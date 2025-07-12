# Launchbox MCP Server

This repository provides tools for generating LaunchBox playlists using natural language queries, as well as other utilities for managing and analyzing your game collection. It is designed to help you automate tasks related to LaunchBox game management or just simply talk to your launchbox.

## Features
- **Natural Language Playlist Generation:** Create LaunchBox-compatible playlists by describing what you want (e.g., "best platformers for SNES and Genesis").
- **Genre and Platform Filtering:** Ensures playlists only include games matching your criteria.
- **Cache and Config Management:** Uses local cache and config files for efficient operation.

## Requirements
- Python 3.7+
- (Optional) LaunchBox installed on your system (for playlist import)
- Recommended: Install dependencies listed in your project (if any)

## Setup
1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/mcp.git
   cd mcp
   ```
2. **(Optional) Create a virtual environment:**
   ```sh
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```
3. **Install dependencies:**
   If you have a `requirements.txt`, install with:
   ```sh
   pip install -r requirements.txt
   ```
   (If not, install any required packages as needed.)


## MCP Tools

The following tools are provided by the MCP server for managing and analyzing your LaunchBox collection:

### 1. `search_games(query, limit=10)`
Searches your LaunchBox game collection for games matching the given query (supports fuzzy/semantic search). Returns a list of matching games.

### 2. `get_game_recommendations(count=5, genre=None, platform=None)`
Recommends unplayed games from your collection, optionally filtered by genre and/or platform. Useful for discovering new games to play.

### 3. `analyze_collection(analysis_type="overview")`
Analyzes your game collection. Supported analysis types:
- `overview`: Stats like total games, platforms, completion rate, genre/platform breakdown, etc.
- `duplicates`: Finds duplicate games by title.
- `unplayed`: Lists unplayed games, grouped by platform/genre.
- `completion`: Lists completed games, grouped by platform/genre.

### 4. `create_playlist(name, criteria, save=False)`
Creates a playlist based on criteria (e.g., "unplayed", "favorite", "completed", "short", "rpg", "recent"). Returns the playlist data and can optionally save it.

### 5. `create_playlist_from_natural_input(natural_input, playlist_name=None, max_results=100)`
Generates a LaunchBox-compatible playlist XML file from a natural language description (e.g., "best platformers for SNES and Genesis"). Uses AI/semantic search to select games and formats the playlist for LaunchBox compatibility.

### 6. `update_game_metadata(game_id, updates)`
Updates metadata for a specific game in your collection (e.g., title, genre, play count).

### 7. `get_game_details(game_id=None, title=None)`
Fetches detailed information about a game, either by its ID or by searching for its title.

### 8. `get_platform_games(platform)`
Lists all games for a specific platform, including stats on unplayed and completed games.

### 9. `get_gaming_stats()`
Provides statistics about your collection, such as most played games, recently played, completion/favorite rates, and breakdowns by platform and genre.

### 10. `refresh_data()`
Refreshes the internal cache and data from your LaunchBox collection, ensuring all information is up to date.

### 11. `get_smart_recommendations(count=5)`
Suggests unplayed games based on your favorite genres and recent play history, with reasons for each recommendation.


## Usage
### Generating a Playlist
1. **Open your MCP Host and configure**
   ```python
    "mcpServers": {
        "Launchbox": {
            "command": "uv",
            "args": [
                "run",
                "--with",
                "mcp",
                "mcp",
                "run",
                "C:\\location\\launchbox.py"
            ]
        }
    }
   ```


## Project Structure
- `launchbox.py` — Main script for playlist generation and LaunchBox utilities
- `launchbox_config.json` — Configuration file

## Video of mcp server in action

https://www.youtube.com/watch?v=QIqef_1Wpk0

## Notes
- Playlists are formatted to match LaunchBox's requirements (field order, indentation, self-closing tags for empty fields).
- If you encounter issues with LaunchBox deleting playlists, ensure the XML formatting matches that of manually created playlists.
- For best results, use clear and specific queries when generating playlists.

## License
MIT License
