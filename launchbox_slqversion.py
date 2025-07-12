#!/usr/bin/env python3
"""
LaunchBox MCP Server Implementation using FastMCP
A Model Context Protocol server for LaunchBox game collection management
"""
import os
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from xml.etree import ElementTree as ET

import os
print(f"[BOOT] Python process CWD: {os.getcwd()}")
import logging
logging.basicConfig(level=logging.INFO)
logging.info(f"[BOOT] Python process CWD: {os.getcwd()}")

# --- LOGGING SETUP (ALWAYS AT TOP) ---
import os
import sys
DEBUG = '--debug' in sys.argv
log_file_path = os.path.join(os.path.dirname(__file__), 'mcp_debug.log')
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.WARNING,
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)
if DEBUG:
    print("[DEBUG] Debug mode enabled.")
    logger.debug("Debug mode enabled.")
# --- END LOGGING SETUP ---

# FastMCP dependencies
from mcp.server.fastmcp import FastMCP

# Additional dependencies for full functionality
# pip install fastmcp xmltodict pandas

logger = logging.getLogger(__name__)

@dataclass
class GameInfo:
    """Represents a game in the LaunchBox collection"""
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
        
    def parse_platforms(self) -> Dict[str, dict]:
        """Parse Platforms.xml"""
        platforms_file = self.data_path / "Platforms.xml"
        if not platforms_file.exists():
            logger.warning(f"Platforms.xml not found at {platforms_file}")
            return {}
            
        try:
            tree = ET.parse(platforms_file)
            root = tree.getroot()
            
            platforms = {}
            for platform in root.findall('Platform'):
                name = platform.find('Name')
                if name is not None:
                    folder_elem = platform.find('Folder')
                    emulator_elem = platform.find('Emulator')
                    category_elem = platform.find('Category')
                    platforms[name.text] = {
                        'name': name.text if name is not None and name.text is not None else '',
                        'folder': folder_elem.text if folder_elem is not None and folder_elem.text is not None else '',
                        'emulator': emulator_elem.text if emulator_elem is not None and emulator_elem.text is not None else '',
                        'category': category_elem.text if category_elem is not None and category_elem.text is not None else ''
                    }
            
            self.platforms = platforms
            return platforms
            
        except Exception as e:
            logger.error(f"Error parsing platforms: {e}")
            return {}
    
    def parse_games(self) -> Dict[str, GameInfo]:
        """Parse all game XML files, including those in Playlists/ and other subdirectories"""
        games = {}
        # Recursively find all XML files except Platforms.xml and Playlists.xml in the root
        xml_files = list(self.data_path.glob("*.xml"))
        # Also include XML files in Playlists/ and other subdirectories
        for subdir in self.data_path.iterdir():
            if subdir.is_dir():
                xml_files.extend(subdir.glob("*.xml"))

        for xml_file in xml_files:
            if xml_file.name in ["Platforms.xml", "Playlists.xml"] and xml_file.parent == self.data_path:
                continue
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                # For each <Game> in the XML
                for game in root.findall('Game'):
                    game_id = game.find('ID')
                    title = game.find('Title')
                    # Use <Platform> tag if present, otherwise fallback to filename
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
                        # Parse tags
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
        """Update a game's metadata in its XML file and save. Always update in the file where the game was found."""
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
                        # Only update fields, do not move to another file
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
        """Helper to safely get text from XML element"""
        child = element.find(tag)
        return child.text if child is not None and child.text else default

class LaunchBoxMCPServer:
    """Main MCP Server for LaunchBox using FastMCP"""
    
    def __init__(self, config_path: str = "launchbox_config.json"):
        self.config = self._load_config(config_path)
        self.parser = LaunchBoxParser(self.config.get("data_path", ""))
        # Always resolve DB path relative to script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cache_path = self.config.get("cache_path", "launchbox_cache.db")
        if not os.path.isabs(cache_path):
            cache_path = os.path.join(script_dir, cache_path)
        self.db_path = Path(cache_path)
        self._setup_database()
        
        # Initialize FastMCP
        self.mcp = FastMCP("LaunchBox MCP Server")
        self._setup_resources()
        self._setup_tools()
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from JSON file"""
        import os
        # Always resolve config path relative to script directory if not absolute
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
                "cache_path": "launchbox_cache.db"
            }
    
    def _setup_database(self):
        """Initialize SQLite cache database"""
        logger.info(f"Setting up SQLite database at {self.db_path}")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS games (
                id TEXT PRIMARY KEY,
                title TEXT,
                platform TEXT,
                developer TEXT,
                publisher TEXT,
                release_date TEXT,
                genre TEXT,
                rating TEXT,
                play_count INTEGER,
                last_played TEXT,
                file_path TEXT,
                notes TEXT,
                completed BOOLEAN,
                favorite BOOLEAN,
                tags TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS search_cache (
                query TEXT PRIMARY KEY,
                results TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database setup complete.")
    
    def _setup_resources(self):
        """Setup FastMCP resources"""
        
        @self.mcp.resource("launchbox://games")
        async def get_games() -> str:
            """Get complete game collection"""
            await self._refresh_data()
            import sqlite3
            logger.info("Resource: get_games from DB")
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM games")
            col_names = [desc[0] for desc in cursor.description]
            games = [dict(zip(col_names, row)) for row in cursor.fetchall()]
            conn.close()
            return json.dumps({
                "games": games,
                "total_count": len(games),
                "last_updated": datetime.now().isoformat()
            }, indent=2)
        
        @self.mcp.resource("launchbox://platforms")
        async def get_platforms() -> str:
            """Get gaming platforms"""
            await self._refresh_data()
            import sqlite3
            logger.info("Resource: get_platforms from DB")
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT platform FROM games")
            platforms = [row[0] for row in cursor.fetchall()]
            conn.close()
            return json.dumps({
                "platforms": platforms,
                "total_count": len(platforms),
                "last_updated": datetime.now().isoformat()
            }, indent=2)
        
        @self.mcp.resource("launchbox://analytics")
        async def get_analytics() -> str:
            """Get collection analytics"""
            await self._refresh_data()
            logger.info("Resource: get_analytics from DB")
            analytics = await self._analyze_collection("overview")
            return json.dumps(analytics, indent=2)
    
    def _setup_tools(self):
        """Setup FastMCP tools"""
        
        @self.mcp.tool()
        async def search_games(query: str, limit: int = 10) -> dict:
            """Search games using natural language queries
            
            Args:
                query: Search query (e.g., "RPGs from the 90s", "unplayed fighting games")
                limit: Maximum number of results to return
            """
            await self._refresh_data()
            return await self._search_games(query, limit)
        
        @self.mcp.tool()
        async def get_game_recommendations(count: int = 5, genre: Optional[str] = None, platform: Optional[str] = None) -> dict:
            """Get personalized game recommendations
            
            Args:
                count: Number of recommendations to return
                genre: Filter by specific genre (optional)
                platform: Filter by specific platform (optional)
            """
            await self._refresh_data()
            return await self._get_recommendations(count, genre, platform)
        
        @self.mcp.tool()
        async def analyze_collection(analysis_type: str = "overview") -> dict:
            """Analyze game collection and provide insights
            
            Args:
                analysis_type: Type of analysis - "overview", "duplicates", "unplayed", "completion", "health"
            """
            await self._refresh_data()
            return await self._analyze_collection(analysis_type)
        
        @self.mcp.tool()
        async def create_playlist(name: str, criteria: str, save: bool = False) -> dict:
            """Create smart playlists based on criteria
            
            Args:
                name: Name for the playlist
                criteria: Selection criteria (e.g., "unplayed RPGs", "favorite games")
                save: Whether to save the playlist to LaunchBox
            """
            await self._refresh_data()
            return await self._create_playlist(name, criteria, save)
        
        @self.mcp.tool()
        async def update_game_metadata(game_id: str, updates: dict) -> dict:
            """Update game information and metadata
            
            Args:
                game_id: ID of the game to update
                updates: Dictionary of fields to update
            """
            await self._refresh_data()
            return await self._update_game_metadata(game_id, updates)
        
        @self.mcp.tool()
        async def get_game_details(game_id: Optional[str] = None, title: Optional[str] = None) -> dict:
            """Get detailed information about a specific game (DB)"""
            await self._refresh_data()
            logger.info(f"Tool: get_game_details from DB: game_id={game_id}, title={title}")
            return await self._get_game_details(game_id, title)
        
        @self.mcp.tool()
        async def get_platform_games(platform: str) -> dict:
            """Get all games for a specific platform (DB)"""
            await self._refresh_data()
            logger.info(f"Tool: get_platform_games from DB: platform={platform}")
            return await self._get_platform_games(platform)
        
        @self.mcp.tool()
        async def get_gaming_stats() -> dict:
            """Get comprehensive gaming statistics and insights (DB)"""
            await self._refresh_data()
            logger.info("Tool: get_gaming_stats from DB")
            return await self._get_gaming_stats()
        
        @self.mcp.tool()
        async def refresh_data() -> dict:
            """Manually refresh LaunchBox XML data and cache."""
            await self._refresh_data()
            return {"status": "Data refreshed", "timestamp": datetime.now().isoformat()}
        
        @self.mcp.tool()
        async def get_smart_recommendations(count: int = 5) -> dict:
            """Get smart game recommendations based on play history, favorite genres, and similar games.
            Args:
                count: Number of recommendations to return
            Returns:
                List of recommended games with reasons
            """
            await self._refresh_data()
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            # 1. Get most played genres
            cursor.execute("SELECT genre, SUM(play_count) as total FROM games WHERE play_count > 0 GROUP BY genre ORDER BY total DESC")
            genre_rows = cursor.fetchall()
            favorite_genres = [row[0] for row in genre_rows if row[0]]
            # 2. Get recently played games
            cursor.execute("SELECT * FROM games WHERE last_played IS NOT NULL AND last_played != '' ORDER BY last_played DESC LIMIT 5")
            col_names = [desc[0] for desc in cursor.description]
            recent_games = [dict(zip(col_names, row)) for row in cursor.fetchall()]
            # 3. Recommend unplayed games in favorite genres, or similar to recent games
            recommendations = []
            reasons = []
            # Recommend unplayed games in favorite genres
            for genre in favorite_genres:
                cursor.execute("SELECT * FROM games WHERE play_count = 0 AND genre = ? LIMIT ?", (genre, count))
                for row in cursor.fetchall():
                    game = dict(zip(col_names, row))
                    if game not in recommendations:
                        recommendations.append(game)
                        reasons.append(f"You like {genre} games")
                    if len(recommendations) >= count:
                        break
                if len(recommendations) >= count:
                    break
            # If not enough, recommend unplayed games similar to recent games (by genre)
            if len(recommendations) < count and recent_games:
                for recent in recent_games:
                    genre = recent.get('genre')
                    if genre:
                        cursor.execute("SELECT * FROM games WHERE play_count = 0 AND genre = ? LIMIT ?", (genre, count - len(recommendations)))
                        for row in cursor.fetchall():
                            game = dict(zip(col_names, row))
                            if game not in recommendations:
                                recommendations.append(game)
                                reasons.append(f"Because you recently played a {genre} game")
                            if len(recommendations) >= count:
                                break
                    if len(recommendations) >= count:
                        break
            # If still not enough, recommend any unplayed games
            if len(recommendations) < count:
                cursor.execute("SELECT * FROM games WHERE play_count = 0 LIMIT ?", (count - len(recommendations),))
                for row in cursor.fetchall():
                    game = dict(zip(col_names, row))
                    if game not in recommendations:
                        recommendations.append(game)
                        reasons.append("Try something new!")
                    if len(recommendations) >= count:
                        break
            conn.close()
            # Attach reasons to each recommendation
            recs_with_reasons = []
            for i, game in enumerate(recommendations):
                recs_with_reasons.append({"game": game, "reason": reasons[i] if i < len(reasons) else ""})
            return {
                "recommendations": recs_with_reasons,
                "favorite_genres": favorite_genres,
                "recent_games": [g['title'] for g in recent_games],
                "count": len(recs_with_reasons)
            }
    
    async def _refresh_data(self):
        """Refresh data from LaunchBox XML files"""
        self.parser.parse_platforms()
        self.parser.parse_games()
        await self._update_cache()
    
    async def _update_cache(self):
        """Update SQLite cache with latest data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Clear existing data
        cursor.execute("DELETE FROM games")
        
        # Insert updated data
        for game in self.parser.games.values():
            cursor.execute('''
                INSERT INTO games VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                game.id, game.title, game.platform, game.developer, game.publisher,
                game.release_date, game.genre, game.rating, game.play_count,
                game.last_played, game.file_path, game.notes, game.completed,
                game.favorite, json.dumps(game.tags), datetime.now()
            ))
        
        conn.commit()
        conn.close()
    
    async def _search_games(self, query: str, limit: int = 10) -> dict:
        """Search games using natural language (now via SQLite DB)"""
        logger.info(f"Searching games in DB for query: '{query}' with limit {limit}")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        query_lower = query.lower()
        filters = []
        sql = "SELECT * FROM games WHERE 1=1"
        params = []
        # Parse query for specific terms
        is_unplayed = "unplayed" in query_lower or "haven't played" in query_lower
        is_completed = "completed" in query_lower or "finished" in query_lower
        is_favorite = "favorite" in query_lower
        if is_unplayed:
            sql += " AND play_count = 0"
        if is_completed:
            sql += " AND completed = 1"
        if is_favorite:
            sql += " AND favorite = 1"
        # Title/genre/platform/developer/publisher/tags/notes search
        like_fields = ["title", "genre", "platform", "developer", "publisher", "tags", "notes"]
        like_clauses = []
        for field in like_fields:
            like_clauses.append(f"LOWER({field}) LIKE ?")
            params.append(f"%{query_lower}%")
        sql += " AND (" + " OR ".join(like_clauses) + ")"
        sql += " ORDER BY play_count DESC LIMIT ?"
        params.append(limit)
        logger.debug(f"SQL: {sql} | Params: {params}")
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        # Map DB rows to dicts
        col_names = [desc[0] for desc in cursor.description]
        results = []
        for row in rows:
            game = dict(zip(col_names, row))
            # Parse tags from JSON
            if game.get("tags"):
                try:
                    game["tags"] = json.loads(game["tags"])
                except:
                    game["tags"] = []
            results.append({"game": game, "score": game["play_count"], "match_reasons": []})
        conn.close()
        return {
            "query": query,
            "results": results,
            "total_found": len(results),
            "filters_applied": {
                "unplayed_only": is_unplayed,
                "completed_only": is_completed,
                "favorites_only": is_favorite
            }
        }
    
    def _get_match_reasons(self, game: GameInfo, query: str) -> List[str]:
        """Get reasons why a game matches the search query"""
        reasons = []
        
        if query in game.title.lower():
            reasons.append("Title match")
        if game.genre and query in game.genre.lower():
            reasons.append("Genre match")
        if query in game.platform.lower():
            reasons.append("Platform match")
        if game.developer and query in game.developer.lower():
            reasons.append("Developer match")
        
        return reasons
    
    async def _get_recommendations(self, count: int, genre: Optional[str] = None, platform: Optional[str] = None) -> dict:
        """Get personalized game recommendations (via SQLite DB)"""
        logger.info(f"Getting recommendations from DB: count={count}, genre={genre}, platform={platform}")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        sql = "SELECT * FROM games WHERE play_count = 0"
        params = []
        if genre:
            sql += " AND LOWER(genre) LIKE ?"
            params.append(f"%{genre.lower()}%")
        if platform:
            sql += " AND LOWER(platform) = ?"
            params.append(platform.lower())
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        col_names = [desc[0] for desc in cursor.description]
        games = [dict(zip(col_names, row)) for row in rows]
        # Score games (simple: by rating if available)
        for g in games:
            try:
                g["score"] = float(g["rating"]) * 10 if g["rating"] else 0
            except:
                g["score"] = 0
        games.sort(key=lambda g: g["score"], reverse=True)
        recommendations = games[:count]
        conn.close()
        return {
            "recommendations": recommendations,
            "criteria": {"genre": genre, "platform": platform, "count": count},
            "total_unplayed": len(games)
        }
    
    async def _analyze_collection(self, analysis_type: str) -> dict:
        """Analyze game collection (via SQLite DB)"""
        logger.info(f"Analyzing collection in DB: {analysis_type}")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        if analysis_type == "overview":
            cursor.execute("SELECT COUNT(*) FROM games")
            total_games = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(DISTINCT platform) FROM games")
            total_platforms = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM games WHERE completed = 1")
            completed_games = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM games WHERE favorite = 1")
            favorite_games = cursor.fetchone()[0]
            cursor.execute("SELECT SUM(play_count) FROM games")
            total_playtime = cursor.fetchone()[0] or 0
            cursor.execute("SELECT COUNT(*) FROM games WHERE play_count = 0")
            unplayed_games = cursor.fetchone()[0]
            completion_rate = (completed_games / total_games * 100) if total_games else 0
            # Platform/genre breakdown
            cursor.execute("SELECT platform, COUNT(*) FROM games GROUP BY platform ORDER BY COUNT(*) DESC")
            platform_distribution = dict(cursor.fetchall())
            cursor.execute("SELECT genre, COUNT(*) FROM games GROUP BY genre ORDER BY COUNT(*) DESC LIMIT 10")
            genre_distribution = dict(cursor.fetchall())
            conn.close()
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
            duplicates = []
            title_groups = {}
            
            # Group games by similar titles
            for game in self.parser.games.values():
                title_key = game.title.lower().strip()
                if title_key not in title_groups:
                    title_groups[title_key] = []
                title_groups[title_key].append(game)
            
            # Find duplicates
            for title, game_list in title_groups.items():
                if len(game_list) > 1:
                    duplicates.append({
                        "title": title,
                        "games": [asdict(g) for g in game_list],
                        "platforms": [g.platform for g in game_list]
                    })
            
            return {
                "duplicates": duplicates,
                "total_duplicate_groups": len(duplicates),
                "total_duplicate_games": sum(len(d["games"]) for d in duplicates)
            }
        
        elif analysis_type == "unplayed":
            unplayed = [g for g in self.parser.games.values() if g.play_count == 0]
            return {
                "unplayed_games": [asdict(g) for g in unplayed],
                "count": len(unplayed),
                "percentage": len(unplayed) / len(self.parser.games.values()) * 100 if self.parser.games.values() else 0,
                "by_platform": self._group_by_platform(unplayed),
                "by_genre": self._group_by_genre(unplayed)
            }
        
        elif analysis_type == "completion":
            completed = [g for g in self.parser.games.values() if g.completed]
            return {
                "completed_games": [asdict(g) for g in completed],
                "count": len(completed),
                "completion_rate": len(completed) / len(self.parser.games.values()) * 100 if self.parser.games.values() else 0,
                "by_platform": self._group_by_platform(completed),
                "by_genre": self._group_by_genre(completed)
            }
        
        else:
            return {"error": f"Unknown analysis type: {analysis_type}"}
    
    def _group_by_platform(self, games: List[GameInfo]) -> dict:
        """Group games by platform"""
        platform_groups = {}
        for game in games:
            if game.platform not in platform_groups:
                platform_groups[game.platform] = []
            platform_groups[game.platform].append(asdict(game))
        return platform_groups
    
    def _group_by_genre(self, games: List[GameInfo]) -> dict:
        """Group games by genre"""
        genre_groups = {}
        for game in games:
            if game.genre:
                genres = [g.strip() for g in game.genre.split(',')]
                for genre in genres:
                    if genre not in genre_groups:
                        genre_groups[genre] = []
                    genre_groups[genre].append(asdict(game))
        return genre_groups
    
    async def _get_platform_breakdown(self) -> dict:
        """Get platform distribution (via SQLite DB)"""
        logger.info("Getting platform breakdown from DB")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT platform, COUNT(*) FROM games GROUP BY platform ORDER BY COUNT(*) DESC")
        result = dict(cursor.fetchall())
        conn.close()
        return result
    
    async def _get_genre_breakdown(self) -> dict:
        """Get genre distribution (via SQLite DB)"""
        logger.info("Getting genre breakdown from DB")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT genre, COUNT(*) FROM games GROUP BY genre ORDER BY COUNT(*) DESC LIMIT 10")
        result = dict(cursor.fetchall())
        conn.close()
        return result
    
    async def _get_favorite_genres(self) -> List[str]:
        """Get user's favorite genres based on play history (via SQLite DB)"""
        logger.info("Getting favorite genres from DB")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT genre, SUM(play_count) as total FROM games WHERE play_count > 0 GROUP BY genre ORDER BY total DESC")
        genres = [row[0] for row in cursor.fetchall() if row[0]]
        conn.close()
        return genres
    
    async def _get_popular_platforms(self) -> List[str]:
        """Get user's most played platforms (via SQLite DB)"""
        logger.info("Getting popular platforms from DB")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT platform, SUM(play_count) as total FROM games WHERE play_count > 0 GROUP BY platform ORDER BY total DESC")
        platforms = [row[0] for row in cursor.fetchall() if row[0]]
        conn.close()
        return platforms
    
    async def _create_playlist(self, name: str, criteria: str, save: bool = False) -> dict:
        """Create a smart playlist (via SQLite DB)"""
        logger.info(f"Creating playlist in DB: name={name}, criteria={criteria}, save={save}")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        criteria_lower = criteria.lower()
        sql = "SELECT * FROM games WHERE 1=1"
        params = []
        if "unplayed" in criteria_lower:
            sql += " AND play_count = 0"
        elif "favorite" in criteria_lower:
            sql += " AND favorite = 1"
        elif "completed" in criteria_lower:
            sql += " AND completed = 1"
        elif "short" in criteria_lower:
            sql += " AND (LOWER(genre) LIKE '%arcade%' OR LOWER(genre) LIKE '%puzzle%' OR LOWER(genre) LIKE '%racing%' OR LOWER(genre) LIKE '%fighting%')"
        elif "rpg" in criteria_lower:
            sql += " AND LOWER(genre) LIKE '%rpg%'"
        # Add recent filter
        if "recent" in criteria_lower:
            sql += " AND (last_played IS NOT NULL AND last_played != '')"
        cursor.execute(sql, params)
        col_names = [desc[0] for desc in cursor.description]
        games = [dict(zip(col_names, row)) for row in cursor.fetchall()]
        conn.close()
        playlist = {
            "name": name,
            "criteria": criteria,
            "games": games,
            "count": len(games),
            "created": datetime.now().isoformat(),
            "saved": save
        }
        if save:
            logger.info(f"Would save playlist '{name}' with {len(games)} games (not implemented)")
        return playlist
    
    async def _update_game_metadata(self, game_id: str, updates: dict) -> dict:
        """Update game metadata (via SQLite DB and XML)"""
        logger.info(f"Updating game metadata in DB: game_id={game_id}, updates={updates}")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM games WHERE id = ?", (game_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return {"error": f"Game {game_id} not found"}
        col_names = [desc[0] for desc in cursor.description]
        game = dict(zip(col_names, row))
        updated_fields = []
        for field, value in updates.items():
            if field == "tags" and isinstance(value, list):
                value = json.dumps(value)
            cursor.execute(f"UPDATE games SET {field} = ? WHERE id = ?", (value, game_id))
            updated_fields.append({"field": field, "old_value": game.get(field), "new_value": value})
        conn.commit()
        conn.close()
        # Also update XML
        xml_result = self.parser.update_game_in_xml(game_id, updates)
        if not xml_result:
            logger.warning(f"Failed to update XML for game {game_id}")
        return {"game_id": game_id, "updated_fields": updated_fields, "xml_updated": xml_result}
    
    def _generate_analytics(self) -> dict:
        """Generate comprehensive analytics"""
        games = list(self.parser.games.values())
        
        return {
            "collection_overview": {
                "total_games": len(games),
                "total_platforms": len(self.parser.platforms),
                "completion_rate": len([g for g in games if g.completed]) / len(games) * 100 if games else 0,
                "favorite_percentage": len([g for g in games if g.favorite]) / len(games) * 100 if games else 0
            },
            "platform_distribution": self._get_platform_breakdown(),
            "genre_distribution": self._get_genre_breakdown(),
            "play_statistics": {
                "most_played_game": max(games, key=lambda g: g.play_count).title if games else None,
                "average_play_count": sum(g.play_count for g in games) / len(games) if games else 0,
                "unplayed_count": len([g for g in games if g.play_count == 0]),
                "completed_count": len([g for g in games if g.completed])
            },
            "generated_at": datetime.now().isoformat()
        }
    
    async def _get_game_details(self, game_id: Optional[str] = None, title: Optional[str] = None) -> dict:
        """Get detailed information about a specific game (DB)"""
        logger.info(f"Getting game details from DB: game_id={game_id}, title={title}")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if game_id:
            cursor.execute("SELECT * FROM games WHERE id = ?", (game_id,))
            row = cursor.fetchone()
            if row:
                col_names = [desc[0] for desc in cursor.description]
                game_data = dict(zip(col_names, row))
                # Parse tags from JSON
                if game_data.get("tags"):
                    try:
                        game_data["tags"] = json.loads(game_data["tags"])
                    except:
                        game_data["tags"] = []
                return {"game": game_data}
            else:
                conn.close()
                return {"error": f"Game with ID {game_id} not found"}
        
        elif title:
            # Search by title
            cursor.execute("SELECT * FROM games WHERE LOWER(title) LIKE ?", (f"%{title.lower()}%",))
            rows = cursor.fetchall()
            if rows:
                col_names = [desc[0] for desc in cursor.description]
                games_data = [dict(zip(col_names, row)) for row in rows]
                # Parse tags from JSON
                for g in games_data:
                    if g.get("tags"):
                        try:
                            g["tags"] = json.loads(g["tags"])
                        except:
                            g["tags"] = []
                return {"games": games_data}
            else:
                conn.close()
                return {"error": f"No games found matching title '{title}'"}
        
        else:
            conn.close()
            return {"error": "Either game_id or title must be provided"}
    
    async def _get_platform_games(self, platform: str) -> dict:
        """Get all games for a specific platform (DB)"""
        logger.info(f"Getting platform games from DB: platform={platform}")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        platform_games = [g for g in self.parser.games.values() if g.platform.lower() == platform.lower()]
        
        return {
            "platform": platform,
            "games": [asdict(g) for g in platform_games],
            "total_count": len(platform_games),
            "unplayed_count": len([g for g in platform_games if g.play_count == 0]),
            "completed_count": len([g for g in platform_games if g.completed])
        }
    
    async def _get_gaming_stats(self) -> dict:
        """Get comprehensive gaming statistics and insights (DB)"""
        logger.info("Getting gaming stats from DB")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        games = list(self.parser.games.values())
        
        # Most played games
        most_played = sorted(games, key=lambda g: g.play_count, reverse=True)[:10]
        
        # Recently played
        recently_played = [g for g in games if g.last_played]
        recently_played.sort(key=lambda g: g.last_played, reverse=True)
        
        # Completion stats
        completed_games = [g for g in games if g.completed]
        favorite_games = [g for g in games if g.favorite]
        
        return {
            "total_games": len(games),
            "most_played": [{"title": g.title, "platform": g.platform, "play_count": g.play_count} 
                           for g in most_played[:5]],
            "recently_played": [{"title": g.title, "platform": g.platform, "last_played": g.last_played} 
                               for g in recently_played[:5]],
            "completion_stats": {
                "total_completed": len(completed_games),
                "completion_rate": len(completed_games) / len(games) * 100 if games else 0,
                "total_favorites": len(favorite_games),
                "favorite_rate": len(favorite_games) / len(games) * 100 if games else 0
            },
            "platform_breakdown": await self._get_platform_breakdown(),
            "genre_breakdown": await self._get_genre_breakdown()
        }
    
    def get_server(self):
        """Get the FastMCP server instance"""
        return self.mcp

# Expose the FastMCP server instance as a global variable for dev tools
server = LaunchBoxMCPServer().get_server()

def main():
    """Main entry point"""
    import sys
    import os

    cwd = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config_path = os.path.join(script_dir, "launchbox_config.json")
    config_path = sys.argv[1] if (len(sys.argv) > 1 and not sys.argv[1].startswith('--')) else default_config_path

    if DEBUG:
        logger.debug(f"Current working directory: {cwd}")
        logger.debug(f"Script directory: {script_dir}")
        logger.debug(f"Attempting to load config from: {config_path}")
        print(f"Current working directory: {cwd}")
        print(f"Script directory: {script_dir}")
        print(f"Attempting to load config from: {config_path}")
        print(f"Files in script directory ({script_dir}): {os.listdir(script_dir)}")
        logger.debug(f"Files in script directory ({script_dir}): {os.listdir(script_dir)}")

    # Check if config file exists
    if not os.path.isfile(config_path):
        logger.error(f"Config file not found at: {config_path}")
        if DEBUG:
            print(f"ERROR: Config file not found at: {config_path}")
    else:
        if DEBUG:
            logger.debug(f"Config file found at: {config_path}")
            print(f"Config file found at: {config_path}")

    if DEBUG:
        with open("mcp_debug.log", "a", encoding="utf-8") as f:
            f.write(f"[DEBUG] CWD: {cwd}\n[DEBUG] Script dir: {script_dir}\n[DEBUG] Config path: {config_path}\n")

    server = LaunchBoxMCPServer(config_path)
    server.get_server().run()

if __name__ == "__main__":
    main()