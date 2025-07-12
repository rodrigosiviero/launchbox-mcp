# calculator_v2.py
"""
LaunchBox MCP Server Implementation using FastMCP (Vector DB Version)
Uses ChromaDB for vector search and storage instead of SQLite.
"""
import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict, field
from xml.etree import ElementTree as ET
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

@dataclass
class GameInfo:
    id: str
    title: str
    platform: str
    developer: str = ""
    publisher: str = ""
    release_date: str = ""
    genre: str = ""
    rating: str = ""
    play_count: int = 0
    last_played: str = ""
    file_path: str = ""
    notes: str = ""
    completed: bool = False
    favorite: bool = False
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

class LaunchBoxParser:
    """Handles parsing LaunchBox XML data files"""
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.games: Dict[str, GameInfo] = {}
        self.platforms: Dict[str, dict] = {}
        self.playlists: Dict[str, dict] = {}

    def parse_games(self) -> Dict[str, GameInfo]:
        games = {}
        xml_files = list(self.data_path.glob("*.xml"))
        for subdir in self.data_path.iterdir():
            if subdir.is_dir():
                xml_files.extend(subdir.glob("*.xml"))
        for xml_file in xml_files:
            if xml_file.name in ["Platforms.xml", "Playlists.xml"] and xml_file.parent == self.data_path:
                continue
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                for game in root.findall('Game'):
                    game_id = game.find('ID')
                    title = game.find('Title')
                    platform_elem = game.find('Platform')
                    platform = platform_elem.text if platform_elem is not None and platform_elem.text else xml_file.stem
                    if game_id is not None and title is not None:
                        game_info = GameInfo(
                            id=game_id.text or "",
                            title=title.text or "",
                            platform=platform,
                            developer=self._get_text(game, 'Developer'),
                            publisher=self._get_text(game, 'Publisher'),
                            release_date=self._get_text(game, 'ReleaseDate'),
                            genre=self._get_text(game, 'Genre'),
                            rating=self._get_text(game, 'Rating'),
                            play_count=int(self._get_text(game, 'PlayCount', '0')),
                            last_played=self._get_text(game, 'LastPlayed'),
                            file_path=self._get_text(game, 'ApplicationPath'),
                            notes=self._get_text(game, 'Notes'),
                            completed=self._get_text(game, 'Completed').lower() == 'true',
                            favorite=self._get_text(game, 'Favorite').lower() == 'true'
                        )
                        tags = []
                        for tag in game.findall('Tag'):
                            if tag.text:
                                tags.append(tag.text)
                        game_info.tags = tags
                        games[game_info.id] = game_info
            except Exception as e:
                logger.error(f"Error parsing {xml_file}: {e}")
                continue
        self.games = games
        return games

    def update_game_in_xml(self, game_id: str, updates: dict) -> bool:
        xml_files = list(self.data_path.glob("*.xml"))
        for subdir in self.data_path.iterdir():
            if subdir.is_dir():
                xml_files.extend(subdir.glob("*.xml"))
        for xml_file in xml_files:
            if xml_file.name in ["Platforms.xml", "Playlists.xml"] and xml_file.parent == self.data_path:
                continue
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                updated = False
                for game in root.findall('Game'):
                    gid = game.find('ID')
                    if gid is not None and gid.text == game_id:
                        for field, value in updates.items():
                            tag_map = {
                                'id': 'ID',
                                'title': 'Title',
                                'platform': 'Platform',
                                'developer': 'Developer',
                                'publisher': 'Publisher',
                                'release_date': 'ReleaseDate',
                                'genre': 'Genre',
                                'rating': 'Rating',
                                'play_count': 'PlayCount',
                                'last_played': 'LastPlayed',
                                'file_path': 'ApplicationPath',
                                'notes': 'Notes',
                                'completed': 'Completed',
                                'favorite': 'Favorite',
                                'tags': 'Tag',
                            }
                            if field == 'tags':
                                for tag_elem in game.findall('Tag'):
                                    game.remove(tag_elem)
                                if isinstance(value, str):
                                    import json
                                    try:
                                        value = json.loads(value)
                                    except Exception:
                                        value = []
                                for tag in value:
                                    tag_elem = ET.SubElement(game, 'Tag')
                                    tag_elem.text = tag
                            elif field in tag_map:
                                tag_name = tag_map[field]
                                elem = game.find(tag_name)
                                if elem is None:
                                    elem = ET.SubElement(game, tag_name)
                                if isinstance(value, bool):
                                    elem.text = 'true' if value else 'false'
                                else:
                                    elem.text = str(value)
                        updated = True
                if updated:
                    tree.write(xml_file, encoding='utf-8', xml_declaration=True)
                    return True
            except Exception as e:
                logger.error(f"Error updating XML {xml_file}: {e}")
                continue
        return False

    def _get_text(self, element, tag, default=""):
        child = element.find(tag)
        return child.text if child is not None and child.text else default

class LaunchBoxMCPServerV2:
    def __init__(self, config_path: str = "launchbox_config.json"):
        self.config = self._load_config(config_path)
        self.parser = LaunchBoxParser(self.config.get("data_path", ""))
        chroma_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".chroma_data")
        self.chroma_client = PersistentClient(path=chroma_dir)
        self.collection = self.chroma_client.get_or_create_collection("games")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        # self._refresh_data()  # Removed automatic refresh at startup
        self.mcp = FastMCP("LaunchBox MCP Server v2 (ChromaDB)")
        self._setup_resources()
        self._setup_tools()

    def _load_config(self, config_path: str) -> dict:
        import os
        if not os.path.isabs(config_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(script_dir, config_path)
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return {
                "data_path": "C:/Users/username/LaunchBox/Data",
            }

    def _refresh_data(self):
        self.parser.parse_games()
        self._update_chroma()

    def _update_chroma(self):
        # Drop and recreate the collection for a full refresh
        self.chroma_client.delete_collection("games")
        self.collection = self.chroma_client.get_or_create_collection("games")
        for game in self.parser.games.values():
            doc = asdict(game)
            text = self._game_to_text(game)
            embedding = self.embedder.encode(text).tolist()
            self.collection.add(
                ids=[game.id],
                documents=[json.dumps(doc)],
                embeddings=[embedding],
                metadatas=[{"title": game.title}]
            )

    def _game_to_text(self, game: GameInfo) -> str:
        return f"{game.title} {game.platform} {game.developer} {game.publisher} {game.release_date} {game.genre} {game.rating} {game.notes} {' '.join(game.tags)}"

    def _search_games(self, query: str, limit: int = 10) -> dict:
        embedding = self.embedder.encode(query).tolist()
        results = self.collection.query(query_embeddings=[embedding], n_results=limit)
        docs = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0]
        ids = (results.get("ids") or [[]])[0]
        if not docs or not metas or not ids:
            return {
                "query": query,
                "results": [],
                "total_found": 0,
                "filters_applied": {}
            }
        games = []
        for doc, meta, id_ in zip(docs, metas, ids):
            game = json.loads(doc)
            games.append({"game": game, "score": 1.0, "match_reasons": ["vector similarity"]})
        return {
            "query": query,
            "results": games,
            "total_found": len(games),
            "filters_applied": {}
        }

    def _update_game_metadata(self, game_id: str, updates: dict) -> dict:
        # Update in XML
        xml_result = self.parser.update_game_in_xml(game_id, updates)
        # Update in ChromaDB
        game = self.parser.games.get(game_id)
        if not game:
            return {"error": f"Game {game_id} not found"}
        for field, value in updates.items():
            setattr(game, field, value)
        doc = asdict(game)
        text = self._game_to_text(game)
        embedding = self.embedder.encode(text).tolist()
        self.collection.upsert(
            ids=[game.id],
            documents=[json.dumps(doc)],
            embeddings=[embedding],
            metadatas=[{"title": game.title}]
        )
        return {"game_id": game_id, "updated_fields": list(updates.keys()), "xml_updated": xml_result}

    def _setup_resources(self):
        @self.mcp.resource("launchbox://games")
        async def get_games() -> str:
            games = [asdict(g) for g in self.parser.games.values()]
            return json.dumps({
                "games": games,
                "total_count": len(games),
                "last_updated": datetime.now().isoformat()
            }, indent=2)

    def _setup_tools(self):
        @self.mcp.tool()
        async def search_games(query: str, limit: int = 10) -> dict:
            return self._search_games(query, limit)

        @self.mcp.tool()
        async def get_game_recommendations(count: int = 5, genre: Optional[str] = None, platform: Optional[str] = None) -> dict:
            # Recommend unplayed games in genre/platform
            games = [g for g in self.parser.games.values() if g.play_count == 0]
            if genre:
                games = [g for g in games if genre.lower() in (g.genre or '').lower()]
            if platform:
                games = [g for g in games if platform.lower() in (g.platform or '').lower()]
            recommendations = sorted(games, key=lambda g: g.title)[:count]
            return {
                "recommendations": [asdict(g) for g in recommendations],
                "criteria": {"genre": genre, "platform": platform, "count": count},
                "total_unplayed": len(games)
            }

        @self.mcp.tool()
        async def analyze_collection(analysis_type: str = "overview") -> dict:
            games = list(self.parser.games.values())
            if analysis_type == "overview":
                total_games = len(games)
                total_platforms = len(set(g.platform for g in games))
                completed_games = len([g for g in games if g.completed])
                favorite_games = len([g for g in games if g.favorite])
                total_playtime = sum(g.play_count for g in games)
                unplayed_games = len([g for g in games if g.play_count == 0])
                completion_rate = (completed_games / total_games * 100) if total_games else 0
                platform_distribution = {}
                for g in games:
                    platform_distribution[g.platform] = platform_distribution.get(g.platform, 0) + 1
                genre_distribution = {}
                for g in games:
                    for genre in (g.genre or '').split(';'):
                        genre = genre.strip()
                        if genre:
                            genre_distribution[genre] = genre_distribution.get(genre, 0) + 1
                return {
                    "total_games": total_games,
                    "total_platforms": total_platforms,
                    "completed_games": completed_games,
                    "favorite_games": favorite_games,
                    "total_playtime": total_playtime,
                    "unplayed_games": unplayed_games,
                    "completion_rate": completion_rate,
                    "platform_distribution": platform_distribution,
                    "genre_distribution": genre_distribution
                }
            elif analysis_type == "duplicates":
                title_groups = {}
                for g in games:
                    key = g.title.lower().strip()
                    title_groups.setdefault(key, []).append(g)
                duplicates = [
                    {"title": title, "games": [asdict(g) for g in group], "platforms": [g.platform for g in group]}
                    for title, group in title_groups.items() if len(group) > 1
                ]
                return {
                    "duplicates": duplicates,
                    "total_duplicate_groups": len(duplicates),
                    "total_duplicate_games": sum(len(d["games"]) for d in duplicates)
                }
            elif analysis_type == "unplayed":
                unplayed = [g for g in games if g.play_count == 0]
                by_platform = {}
                by_genre = {}
                for g in unplayed:
                    by_platform.setdefault(g.platform, []).append(asdict(g))
                    for genre in (g.genre or '').split(';'):
                        genre = genre.strip()
                        if genre:
                            by_genre.setdefault(genre, []).append(asdict(g))
                return {
                    "unplayed_games": [asdict(g) for g in unplayed],
                    "count": len(unplayed),
                    "percentage": len(unplayed) / len(games) * 100 if games else 0,
                    "by_platform": by_platform,
                    "by_genre": by_genre
                }
            elif analysis_type == "completion":
                completed = [g for g in games if g.completed]
                by_platform = {}
                by_genre = {}
                for g in completed:
                    by_platform.setdefault(g.platform, []).append(asdict(g))
                    for genre in (g.genre or '').split(';'):
                        genre = genre.strip()
                        if genre:
                            by_genre.setdefault(genre, []).append(asdict(g))
                return {
                    "completed_games": [asdict(g) for g in completed],
                    "count": len(completed),
                    "completion_rate": len(completed) / len(games) * 100 if games else 0,
                    "by_platform": by_platform,
                    "by_genre": by_genre
                }
            else:
                return {"error": f"Unknown analysis type: {analysis_type}"}

        @self.mcp.tool()
        async def create_playlist(name: str, criteria: str, save: bool = False) -> dict:
            games = list(self.parser.games.values())
            criteria_lower = criteria.lower()
            if "unplayed" in criteria_lower:
                games = [g for g in games if g.play_count == 0]
            elif "favorite" in criteria_lower:
                games = [g for g in games if g.favorite]
            elif "completed" in criteria_lower:
                games = [g for g in games if g.completed]
            elif "short" in criteria_lower:
                games = [g for g in games if any(x in (g.genre or '').lower() for x in ["arcade", "puzzle", "racing", "fighting"])]
            elif "rpg" in criteria_lower:
                games = [g for g in games if "rpg" in (g.genre or '').lower()]
            if "recent" in criteria_lower:
                games = [g for g in games if g.last_played]
            playlist = {
                "name": name,
                "criteria": criteria,
                "games": [asdict(g) for g in games],
                "count": len(games),
                "created": datetime.now().isoformat(),
                "saved": save
            }
            return playlist

        @self.mcp.tool()
        async def create_playlist_from_natural_input(natural_input: str, playlist_name: Optional[str] = None, max_results: int = 100) -> dict:
            """
            Create a LaunchBox playlist XML from a natural language input using vector search (AI/semantic matching).
            Generates a LaunchBox-compatible playlist file with the correct structure and all required fields.
            Now pretty-prints (indents) the XML to match manual formatting.
            """
            import uuid
            from xml.etree.ElementTree import Element, SubElement, ElementTree
            # Use vector search to find relevant games
            search_results = self._search_games(natural_input, limit=max_results)
            games = [json.loads(r['game']) if isinstance(r['game'], str) else r['game'] for r in search_results.get('results', [])]
            if not playlist_name:
                playlist_name = f"AI_{natural_input.replace(' ', '_')[:40]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            playlist_id = str(uuid.uuid4())
            playlist_root = Element('LaunchBox')
            playlist_elem = SubElement(playlist_root, 'Playlist')
            def add_elem(parent, tag, text=None):
                if text is None or text == '':
                    SubElement(parent, tag)
                else:
                    SubElement(parent, tag).text = str(text)
            add_elem(playlist_elem, 'PlaylistId', playlist_id)
            add_elem(playlist_elem, 'Name', playlist_name)
            add_elem(playlist_elem, 'NestedName', playlist_name)
            add_elem(playlist_elem, 'SortBy', 'StarRating')
            add_elem(playlist_elem, 'Notes')
            add_elem(playlist_elem, 'VideoPath')
            add_elem(playlist_elem, 'ImageType')
            add_elem(playlist_elem, 'Category')
            add_elem(playlist_elem, 'LastGameId')
            add_elem(playlist_elem, 'BigBoxView')
            add_elem(playlist_elem, 'BigBoxTheme')
            add_elem(playlist_elem, 'IncludeWithPlatforms', 'false')
            add_elem(playlist_elem, 'AutoPopulate', 'false')
            add_elem(playlist_elem, 'SortTitle')
            add_elem(playlist_elem, 'IsAutogenerated', 'false')
            add_elem(playlist_elem, 'LocalDbParsed', 'false')
            add_elem(playlist_elem, 'LastSelectedChild')
            add_elem(playlist_elem, 'Developer')
            add_elem(playlist_elem, 'Manufacturer')
            add_elem(playlist_elem, 'Cpu')
            add_elem(playlist_elem, 'Memory')
            add_elem(playlist_elem, 'Graphics')
            add_elem(playlist_elem, 'Sound')
            add_elem(playlist_elem, 'Display')
            add_elem(playlist_elem, 'Media')
            add_elem(playlist_elem, 'MaxControllers')
            add_elem(playlist_elem, 'Folder')
            add_elem(playlist_elem, 'VideosFolder')
            add_elem(playlist_elem, 'FrontImagesFolder')
            add_elem(playlist_elem, 'BackImagesFolder')
            add_elem(playlist_elem, 'ClearLogoImagesFolder')
            add_elem(playlist_elem, 'FanartImagesFolder')
            add_elem(playlist_elem, 'ScreenshotImagesFolder')
            add_elem(playlist_elem, 'BannerImagesFolder')
            add_elem(playlist_elem, 'SteamBannerImagesFolder')
            add_elem(playlist_elem, 'ManualsFolder')
            add_elem(playlist_elem, 'MusicFolder')
            add_elem(playlist_elem, 'ScrapeAs')
            add_elem(playlist_elem, 'AndroidThemeVideoPath')
            add_elem(playlist_elem, 'HideInBigBox', 'false')
            add_elem(playlist_elem, 'DisableAutoImport', 'false')
            for idx, g in enumerate(games):
                game_elem = SubElement(playlist_root, 'PlaylistGame')
                add_elem(game_elem, 'GameId', g.get('id', ''))
                add_elem(game_elem, 'GameTitle', g.get('title', ''))
                if g.get('file_path'):
                    add_elem(game_elem, 'GameFileName', g['file_path'].split('\\')[-1].split('/')[-1])
                else:
                    add_elem(game_elem, 'GameFileName')
                add_elem(game_elem, 'GamePlatform', g.get('platform', ''))
                add_elem(game_elem, 'ManualOrder', str(idx))
            # Pretty-print (indent) the XML
            def indent(elem, level=0):
                i = "\n" + level*"  "
                if len(elem):
                    if not elem.text or not elem.text.strip():
                        elem.text = i + "  "
                    for e in elem:
                        indent(e, level+1)
                    if not e.tail or not e.tail.strip():
                        e.tail = i
                else:
                    if level and (not elem.tail or not elem.tail.strip()):
                        elem.tail = i
            indent(playlist_root)
            playlists_dir = Path(self.parser.data_path) / 'Playlists'
            playlists_dir.mkdir(parents=True, exist_ok=True)
            playlist_path = playlists_dir / f"{playlist_name}.xml"
            tree = ElementTree(playlist_root)
            with open(playlist_path, 'wb') as f:
                f.write(b'<?xml version="1.0" standalone="yes"?>\n')
                tree.write(f, encoding='utf-8', xml_declaration=False, short_empty_elements=True)
            return {
                'playlist_name': playlist_name,
                'playlist_path': str(playlist_path),
                'game_count': len(games),
                'input': natural_input,
                'success': True
            }

        @self.mcp.tool()
        async def update_game_metadata(game_id: str, updates: dict) -> dict:
            return self._update_game_metadata(game_id, updates)

        @self.mcp.tool()
        async def get_game_details(game_id: Optional[str] = None, title: Optional[str] = None) -> dict:
            # Search by ID or title in self.parser.games
            if game_id:
                g = self.parser.games.get(game_id)
                if g:
                    return {"game": asdict(g)}
                else:
                    return {"error": f"Game with ID {game_id} not found"}
            elif title:
                matches = [asdict(g) for g in self.parser.games.values() if title.lower() in g.title.lower()]
                if matches:
                    return {"games": matches}
                else:
                    return {"error": f"No games found matching title '{title}'"}
            else:
                return {"error": "Either game_id or title must be provided"}

        @self.mcp.tool()
        async def get_platform_games(platform: str) -> dict:
            games = [g for g in self.parser.games.values() if g.platform.lower() == platform.lower()]
            return {
                "platform": platform,
                "games": [asdict(g) for g in games],
                "total_count": len(games),
                "unplayed_count": len([g for g in games if g.play_count == 0]),
                "completed_count": len([g for g in games if g.completed])
            }

        @self.mcp.tool()
        async def get_gaming_stats() -> dict:
            games = list(self.parser.games.values())
            most_played = sorted(games, key=lambda g: g.play_count, reverse=True)[:10]
            recently_played = [g for g in games if g.last_played]
            recently_played.sort(key=lambda g: g.last_played, reverse=True)
            completed_games = [g for g in games if g.completed]
            favorite_games = [g for g in games if g.favorite]
            return {
                "total_games": len(games),
                "most_played": [{"title": g.title, "platform": g.platform, "play_count": g.play_count} for g in most_played[:5]],
                "recently_played": [{"title": g.title, "platform": g.platform, "last_played": g.last_played} for g in recently_played[:5]],
                "completion_stats": {
                    "total_completed": len(completed_games),
                    "completion_rate": len(completed_games) / len(games) * 100 if games else 0,
                    "total_favorites": len(favorite_games),
                    "favorite_rate": len(favorite_games) / len(games) * 100 if games else 0
                },
                "platform_breakdown": {p: len([g for g in games if g.platform == p]) for p in set(g.platform for g in games)},
                "genre_breakdown": {genre: len([g for g in games if genre in (g.genre or '')]) for genre in set(sum([g.genre.split(';') for g in games if g.genre], []))}
            }

        @self.mcp.tool()
        async def refresh_data() -> dict:
            self._refresh_data()
            return {"status": "Data refreshed", "timestamp": datetime.now().isoformat()}

        @self.mcp.tool()
        async def get_smart_recommendations(count: int = 5) -> dict:
            games = list(self.parser.games.values())
            # Favorite genres: most played genres
            genre_counts = {}
            for g in games:
                for genre in (g.genre or '').split(';'):
                    genre = genre.strip()
                    if genre:
                        genre_counts[genre] = genre_counts.get(genre, 0) + g.play_count
            favorite_genres = sorted(genre_counts, key=lambda g: genre_counts.get(g, 0), reverse=True)
            recent_games = sorted([g for g in games if g.last_played], key=lambda g: g.last_played, reverse=True)[:5]
            recommendations = []
            reasons = []
            # Recommend unplayed games in favorite genres
            for genre in favorite_genres:
                for g in games:
                    if g.play_count == 0 and genre in (g.genre or '') and g not in recommendations:
                        recommendations.append(g)
                        reasons.append(f"You like {genre} games")
                        if len(recommendations) >= count:
                            break
                if len(recommendations) >= count:
                    break
            # If not enough, recommend unplayed games similar to recent games (by genre)
            if len(recommendations) < count and recent_games:
                for recent in recent_games:
                    for g in games:
                        if g.play_count == 0 and recent.genre in (g.genre or '') and g not in recommendations:
                            recommendations.append(g)
                            reasons.append(f"Because you recently played a {recent.genre} game")
                            if len(recommendations) >= count:
                                break
                    if len(recommendations) >= count:
                        break
            # If still not enough, recommend any unplayed games
            if len(recommendations) < count:
                for g in games:
                    if g.play_count == 0 and g not in recommendations:
                        recommendations.append(g)
                        reasons.append("Try something new!")
                        if len(recommendations) >= count:
                            break
            recs_with_reasons = []
            for i, g in enumerate(recommendations):
                recs_with_reasons.append({"game": asdict(g), "reason": reasons[i] if i < len(reasons) else ""})
            return {
                "recommendations": recs_with_reasons,
                "favorite_genres": favorite_genres,
                "recent_games": [g.title for g in recent_games],
                "count": len(recs_with_reasons)
            }

    def get_server(self):
        return self.mcp

# Usage: server = LaunchBoxMCPServerV2().get_server()
server = LaunchBoxMCPServerV2().get_server()

if __name__ == "__main__":
    # Script to clean GBA game metadata titles
    server = LaunchBoxMCPServerV2()
    parser = server.parser
    updated_count = 0
    for game in parser.games.values():
        if game.platform.lower() in ["gba", "game boy advance"]:
            new_title = game.title.replace(".gba", "").replace("(BR)", "").strip()
            if new_title != game.title:
                print(f"Renaming metadata: '{game.title}' -> '{new_title}'")
                # Update in XML and ChromaDB, including metadata
                server._update_game_metadata(game.id, {"title": new_title})
                # Also update the ChromaDB metadata title field
                server.collection.upsert(
                    ids=[game.id],
                    documents=[json.dumps(asdict(game))],
                    embeddings=[server.embedder.encode(server._game_to_text(game)).tolist()],
                    metadatas=[{"title": new_title}]
                )
                updated_count += 1
    print(f"Updated metadata for {updated_count} GBA game titles.")