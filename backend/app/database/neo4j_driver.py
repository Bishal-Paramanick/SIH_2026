"""Neo4j Database Driver & Connection Manager.

Handles connection pooling, session lifecycle, health check verification,
and parameterized Cypher query execution.
"""

from pathlib import Path
import os
from typing import Any
from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase

# Locate .env robustly relative to backend directory
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=ENV_PATH)

NEO4J_URL = os.getenv("NEO4J_URL", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "bishal005")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")


class Neo4jConnection:
    """Manages connection lifecycle with Neo4j database."""

    def __init__(
        self,
        url: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
    ):
        self.url = url or os.getenv("NEO4J_URL", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "bishal005")
        self.database = database or os.getenv("NEO4J_DATABASE", "neo4j")
        self.driver: Driver | None = None

    def connect(self) -> Driver:
        if self.driver is None:
            self.driver = GraphDatabase.driver(
                self.url, auth=(self.user, self.password)
            )
        return self.driver

    def close(self):
        if self.driver is not None:
            self.driver.close()
            self.driver = None

    def verify_connectivity(self) -> bool:
        """Verifies if database is reachable."""
        try:
            driver = self.connect()
            driver.verify_connectivity()
            return True
        except Exception as e:
            print(f"[Neo4j Error] Connection failed: {e}")
            return False

    def query(
        self,
        cypher_query: str,
        parameters: dict[str, Any] | None = None,
        database: str | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a read/write Cypher query and return list of records as dicts."""
        target_db = database or self.database
        driver = self.connect()
        with driver.session(database=target_db) as session:
            result = session.run(cypher_query, parameters=parameters or {})  # type: ignore
            return [record.data() for record in result]


# Global singleton instance
db = Neo4jConnection()
