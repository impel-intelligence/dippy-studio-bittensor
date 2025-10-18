import json
import logging
import random
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field

from orchestrator.services.database_service import DatabaseService


class Miner(BaseModel):
    uid: int
    network_address: str
    valid: bool
    alpha_stake: int
    capacity: Dict[str, object] = Field(default_factory=dict)
    hotkey: Optional[str] = None


class MinerMetagraphClient(ABC):
    @abstractmethod
    def update_state(
        self,
        state: Dict[str, Miner],
        *,
        block: Optional[int] = None,
        fetched_at: Optional[datetime] = None,
    ) -> None:
        """Persist the provided miner state and record sync metadata."""
        pass

    @abstractmethod
    def dump_state(self):
        pass


class LiveMinerMetagraphClient(MinerMetagraphClient):
    def __init__(self, database_service: DatabaseService) -> None:
        self._instance_id = f"metagraph-{uuid.uuid4().hex}"
        self._database_service = database_service
        self._conn = self._database_service.get_connection()
        self._db_path = str(self._database_service.path)
        # Enforce JSON validity using STRICT table and CHECK json_valid(value)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS miners (
                hotkey TEXT PRIMARY KEY,
                value  TEXT NOT NULL CHECK (json_valid(value))
            ) STRICT
            """
        )
        self._conn.commit()
        # This client no longer handles signing or outbound requests.
        # NOTE: This value was from an older version of the code that was lost in the sauce. 
        self._max_alpha_limit = 1000
        self._last_update: Optional[datetime] = None
        self._last_block: Optional[int] = None

    @property
    def instance_id(self) -> str:
        return self._instance_id

    @property
    def db_path(self) -> str:
        return self._db_path

    @staticmethod
    def validate_state(state: Dict[str, Miner]) -> Dict[str, Miner]:
        logger = logging.getLogger(__name__)
        validated_state = {}
        
        for key, value in state.items():
            
            # If the value is a Miner object, log its attributes and validate network address
            if isinstance(value, Miner):
                
                # Test network address connectivity
                network_valid = False
                
                # Skip request if network address is empty or not parseable
                network_address = value.network_address
                if not network_address or not network_address.strip():
                    network_valid = False
                else:
                    try:
                        # Ensure URL has a scheme
                        parsed = urlparse(network_address)
                        if not parsed.scheme:
                            network_address = f"https://{network_address}/check/{value.hotkey}"
                        # Re-parse to validate the URL is now parseable
                        parsed = urlparse(network_address)
                        if parsed.netloc:                            
                            # Make HTTP GET request with timeout
                            req = Request(network_address, method="GET")
                            with urlopen(req, timeout=10) as response:
                                if response.status == 200:
                                    network_valid = True
                                    
                                    # If check endpoint succeeds, try to fetch capacity
                                    try:
                                        # Construct capacity endpoint URL from parsed scheme/netloc
                                        capacity_url = f"{parsed.scheme}://{parsed.netloc}/capacity"
                                        
                                        # Make request to capacity endpoint
                                        capacity_req = Request(capacity_url, method="GET")
                                        with urlopen(capacity_req, timeout=10) as capacity_response:
                                            if capacity_response.status == 200:
                                                capacity_data = json.loads(capacity_response.read().decode('utf-8'))
                                                # Update the miner's capacity field
                                                value.capacity = capacity_data
                                    except (URLError, HTTPError, json.JSONDecodeError, Exception) as e:
                                        # If capacity endpoint fails, just log and continue
                                        logger.debug(f"Failed to fetch capacity from {capacity_url}: {e}")
                                        # Don't fail the validation if capacity fetch fails
                            
                                    
                    except (URLError, HTTPError, Exception) as e:
                        logger.debug(f"Network address {network_address} unreachable: {e}")

                # Create updated miner with new valid status
                updated_miner = Miner(
                    uid=value.uid,
                    network_address=value.network_address,
                    valid=network_valid,
                    alpha_stake=value.alpha_stake,
                    capacity=value.capacity,
                    hotkey=value.hotkey
                )
                validated_state[key] = updated_miner
            else:
                validated_state[key] = value
        
        return validated_state


    def _miner_to_json(self, miner: Miner) -> str:
        if hasattr(miner, "model_dump_json"):
            return miner.model_dump_json()
        if hasattr(miner, "json"):
            return miner.json()
        return json.dumps(getattr(miner, "dict", lambda: miner)())

    def _json_to_miner(self, payload: str) -> Miner:
        if hasattr(Miner, "model_validate_json"):
            return Miner.model_validate_json(payload)  # type: ignore[attr-defined]
        if hasattr(Miner, "parse_raw"):
            return Miner.parse_raw(payload)  # type: ignore[attr-defined]
        return Miner(**json.loads(payload))

    def update_state(
        self,
        state: Dict[str, Miner],
        *,
        block: Optional[int] = None,
        fetched_at: Optional[datetime] = None,
    ) -> None:
        cur = self._conn.cursor()
        new_keys = set(state.keys())
        cur.execute("SELECT hotkey FROM miners")
        existing_keys = {row[0] for row in cur.fetchall()}
        to_delete = existing_keys - new_keys
        if to_delete:
            cur.executemany("DELETE FROM miners WHERE hotkey = ?", [(k,) for k in to_delete])
        rows = [
            (
                hotkey,
                self._miner_to_json(miner),
            )
            for hotkey, miner in state.items()
        ]
        if rows:
            cur.executemany(
                """
                INSERT INTO miners (hotkey, value)
                VALUES (?, ?)
                ON CONFLICT(hotkey) DO UPDATE SET value=excluded.value
                """,
                rows,
            )
        self._conn.commit()
        self._last_update = fetched_at or datetime.now(timezone.utc)
        self._last_block = block

    def _iter_miners(self, *, valid_only: bool = False):
        cur = self._conn.cursor()
        if valid_only:
            cur.execute(
                """
                SELECT hotkey, value
                FROM miners
                WHERE json_extract(value, '$.valid') IN (1, 'true')
                """
            )
        else:
            cur.execute("SELECT hotkey, value FROM miners")

        for hotkey, payload in cur:
            yield hotkey, self._json_to_miner(payload)

    def dump_state(self) -> Dict[str, Miner]:
        return {hotkey: miner for hotkey, miner in self._iter_miners()}

    def dump_filtered_state(self) -> Dict[str, Miner]:
        return {hotkey: miner for hotkey, miner in self._iter_miners(valid_only=True)}

    def dump_full_state(self) -> Dict[str, Miner]:
        """Return the complete metagraph state without filtering."""
        return self.dump_state()

    # Note: outbound requests and signing were historically handled by a separate network client.

    def last_update(self) -> Optional[datetime]:
        return self._last_update

    def last_block(self) -> Optional[int]:
        return self._last_block

    def fetch_miners(self) -> Dict[str, Miner]:
        return {hotkey: miner for hotkey, miner in self._iter_miners(valid_only=True)}

    def fetch_candidate(self) -> Optional[Miner]:
        """Return a randomly selected valid Miner from current state, weighted by stake

        Only considers miners marked as valid. If no valid miners exist,
        returns None.
        """
        candidates = [miner for _, miner in self._iter_miners(valid_only=True)]
        if not candidates:
            return None

        weights = [
            max(0, min(miner.alpha_stake, self._max_alpha_limit))
            for miner in candidates
        ]

        if sum(weights) <= 0:
            return random.choice(candidates)

        return random.choices(candidates, weights=weights, k=1)[0]
