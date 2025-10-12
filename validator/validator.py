
import base64
import datetime as dt
import os
import time
import torch
import random
import asyncio
import subprocess
import argparse
from typing import Any, Dict, List, Optional, Tuple
import requests
from importlib.metadata import version as pkg_version
from shlex import split
import traceback
import sys
import json
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel, Field
import math

from rich.table import Table
from rich.console import Console

import bittensor as bt
import numpy as np

from bittensor.core.subtensor import Subtensor
from bittensor.core.metagraph import Metagraph


from pathlib import Path


# Project constants

# Validation API hosted by SN Owner
VALIDATION_SERVER = "https://sn11.dippy-bittensor-subnet.com"

# The uid for this subnet.
SUBNET_UID = 11
# The root directory of this project.
ROOT_DIR = Path(__file__).parent.parent

# Miner/Validator model parameters

weights_version_key = 8

alpha = 0.9
temperature = 0.005 * 15

ORCHESTRATION_SERVER = "https://orchestrator.dippy-bittensor-subnet.com"
SUBNET_OWNER_UID = 74
BASE_BURN_RATE = 0.75

class StrEnum(str, Enum):
    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value

    @classmethod
    def from_string(cls, value: str):
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"{value} is not a valid {cls.__name__}")

class StatusEnum(StrEnum):
    QUEUED = "QUEUED"
    PRECHECK = "PRECHECK"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    RUNNING = "RUNNING"
    VALID = "VALID"
    INVALID = "INVALID"
    AUDITED = "AUDITED"

class Scores(BaseModel):
    total_score: float = Field(default=0, description="The total score of the evaluation")
    status: str = Field(default=StatusEnum.QUEUED, description="The current status of the scoring process")

@dataclass
class LocalMetadata:
    """Metadata associated with the local validator instance"""
    commit: str
    btversion: str
    uid: int = 0
    coldkey: str = ""
    hotkey: str = ""

class MinerEntry(BaseModel):
    block: int = Field(default=sys.maxsize, description="The block number")
    hotkey: str = Field(default="", description="The hotkey of the miner")
    invalid: bool = Field(default=False, description="invalidity of determining score")
    total_score: float = Field(default=0)


def assert_registered(wallet: bt.wallet, metagraph: bt.metagraph) -> int:
    """Asserts the wallet is a registered miner and returns the miner's UID."""
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        raise ValueError(
            f"You are not registered. \nUse: \n`btcli s register --netuid {metagraph.netuid}` to register via burn \n or btcli s pow_register --netuid {metagraph.netuid} to register with a proof of work"
        )
    uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
    bt.logging.success(f"You are registered with address: {wallet.hotkey.ss58_address} and uid: {uid}")
    return uid

def local_metadata() -> LocalMetadata:
    """Extract the version as current git commit hash"""
    commit_hash = ""
    try:
        result = subprocess.run(
            split("git rev-parse HEAD"),
            check=True,
            capture_output=True,
            cwd=ROOT_DIR,
        )
        commit = result.stdout.decode().strip()
        assert len(commit) == 40, f"Invalid commit hash: {commit}"
        commit_hash = commit[:8]
    except Exception as e:
        commit_hash = "unknown"

    bittensor_version = pkg_version("bittensor")
    return LocalMetadata(
        commit=commit_hash,
        btversion=bittensor_version,
    )


class Validator:
    @staticmethod
    def config():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--no-verify",
            action="store_true",
            default=False,
            help="Do not verify validator on chain",
        )
        parser.add_argument(
            "--offline",
            action="store_true",
            help="Does not set weights.",
        )
        parser.add_argument(
            "--immediate",
            action="store_true",
            help="Triggers run step immediately",
        )
        parser.add_argument(
            "--local",
            action="store_true",
            help="Toggles for local subtensor",
        )
        parser.add_argument(
            "--network",
            type=str,
            default=None,
            help="Network to use (finney, test, local). Overrides --local flag.",
        )
        parser.add_argument("--netuid", type=int, default=SUBNET_UID, help="The subnet UID.")

        bt.subtensor.add_args(parser)
        bt.logging.add_args(parser)
        bt.wallet.add_args(parser)
        config = bt.config(parser)
        return config

    def __init__(self, local_metadata: LocalMetadata):
        self.config = Validator.config()
        bt.logging(config=self.config)

        # Get bittensor package version
        bt_version = pkg_version("bittensor")
        bt.logging.warning(f"Starting validator with config: {self.config}, bittensor version: {bt_version}")

        # Set verify flag based on --no-verify argument
        self.verify = not self.config.no_verify

        self.wallet = bt.wallet(config=self.config)

        try:
            self.subtensor = Validator.new_subtensor(self.config)
            bt.logging.warning(f"subtensor initialized with Subtensor: {self.subtensor}")
        except Exception as e:
            bt.logging.error(f"could not initialize subtensor: {e}")
            self.subtensor = Validator.new_subtensor(self.config)
            bt.logging.warning(f"subtensor retry initialized with Subtensor(): {self.subtensor}")
        try:
            self.metagraph = self.subtensor.metagraph(netuid=self.config.netuid, lite=False)
        except Exception as e:
            bt.logging.error(f"could not initialize metagraph: {e}")
            raise e

        # Dont check registration status if offline.
        if self.verify:
            self.uid = assert_registered(self.wallet, self.metagraph)

        # === Running args ===
        torch_metagraph = torch.from_numpy(self.metagraph.S).to(torch.float32)

        self.weights = torch.zeros_like(torch_metagraph, dtype=torch.float32)

        torch_consensus = torch.from_numpy(self.metagraph.C).to(torch.float32)
        self.weights.copy_(torch_consensus)

        validator_uid = 0
        if self.verify:
            validator_uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)

        # Set up local metadata for stats collection
        self.local_metadata = LocalMetadata(
            commit=local_metadata.commit,
            btversion=local_metadata.btversion,
            hotkey=self.wallet.hotkey.ss58_address,
            coldkey=self.wallet.coldkeypub.ss58_address,
            uid=validator_uid,
        )
        bt.logging.warning(f"dumping localmetadata: {self.local_metadata}")

    @staticmethod
    def calculate_dynamic_burn_rate(current_block: int) -> float:
        """
        Calculate burn rate - always returns BASE_BURN_RATE (0.75).

        Args:
            current_block: Current blockchain block height (unused)

        Returns:
            float: Fixed burn rate of 0.75
        """
        return BASE_BURN_RATE


    def adjust_weights_for_burn(
        self,
        base_weights: torch.Tensor,
        sorted_uids: List[int],
        burn_rate: float,
        target_uid: int = SUBNET_OWNER_UID
    ) -> torch.Tensor:
        """
        Adjust weight allocations with fixed burn rate.
        75% of total weight goes to target_uid, 25% distributed among miners based on winrate.

        Args:
            base_weights: Original calculated weights (used for proportional distribution)
            sorted_uids: List of UIDs in same order as weights
            burn_rate: Burn rate (always 0.75)
            target_uid: UID to receive burned tokens

        Returns:
            torch.Tensor: Adjusted weights with burn allocation
        """
        try:
            adjusted_weights = torch.zeros_like(base_weights)

            target_idx = None
            try:
                target_idx = sorted_uids.index(target_uid)
            except ValueError:
                bt.logging.warning(f"Target UID {target_uid} not found in current metagraph")
                return base_weights

            # Allocate 75% to target_uid
            adjusted_weights[target_idx] = burn_rate
            
            # Allocate remaining 25% proportionally based on base_weights
            remaining_weight = 1.0 - burn_rate
            base_sum = base_weights.sum()
            
            if base_sum > 0:
                # Distribute remaining weight proportionally, excluding target_uid
                for i, uid in enumerate(sorted_uids):
                    if uid != target_uid:
                        adjusted_weights[i] = base_weights[i] * (remaining_weight / base_sum)

            bt.logging.info(
                f"Applied fixed burn: rate={burn_rate:.6f}, allocated_to_uid={target_uid}, "
                f"remaining_for_miners={remaining_weight:.6f}"
            )

            return adjusted_weights

        except Exception as e:
            bt.logging.error(f"Error adjusting weights for burn: {e}")
            return base_weights


    @staticmethod
    def score_to_winrate(
        miner_registry: Dict[int, MinerEntry],
    ) -> Tuple[Dict[int, int], Dict[int, float]]:

        uids = miner_registry.keys()
        wins = {uid: 0 for uid in uids}
        win_rate = {uid: 0 for uid in uids}
        for i, uid_i in enumerate(uids):
            total_matches = 0
            for j, uid_j in enumerate(uids):
                if i == j:
                    continue

                score_i = miner_registry[uid_i].total_score
                score_j = miner_registry[uid_j].total_score

                wins[uid_i] += 1 if score_i > score_j else 0
                total_matches += 1
            # Calculate win rate for uid i
            win_rate[uid_i] = wins[uid_i] / total_matches if total_matches > 0 else 0

            if miner_registry[uid_i].invalid or miner_registry[uid_i].total_score == 0:
                win_rate[uid_i] = float("-inf")

        return wins, win_rate

    def _reinit_subtensor(self):
        """Re-initializes the subtensor connection."""
        bt.logging.info("Re-initializing subtensor connection.")
        self.subtensor = Validator.new_subtensor(self.config)
        bt.logging.info(f"Subtensor re-initialized: {self.subtensor}")

    def __del__(self):
        if hasattr(self, "subtensor"):
            self.subtensor.close()

    def _event_log(self, msg: str, **kwargs):
        bt.logging.debug(f"Event: {msg} {kwargs}")

    def _with_decoration(self, keypair, payload_to_log: Dict[str, Any]) -> Dict[str, Any]:
        signature = sign_request(
            keypair,
            payload=self.local_metadata.hotkey,
        )
        combined_payload = {
            "signature": signature,
            "event_payload": payload_to_log,
            "commit": str(self.local_metadata.commit),
            "btversion": str(self.local_metadata.btversion),
            "uid": str(self.local_metadata.uid),
            "hotkey": str(self.local_metadata.hotkey),
            "coldkey": str(self.local_metadata.coldkey),
        }
        return combined_payload

    def _remote_log(self, event_data: Dict[str, Any]):
        event_report_endpoint = f"{VALIDATION_SERVER}/event_report"
        try:
            decorated_payload = self._with_decoration(self.wallet.hotkey, event_data)
            response = requests.post(event_report_endpoint, json=decorated_payload)
            response.raise_for_status()
            bt.logging.debug(f"Successfully sent event_report for: {event_data}")
        except Exception as e:
            bt.logging.error(f"Could not remote log event {event_data}: {e}. This error is ok to ignore if you are a validator")

    async def set_weights_with_wait(self, weights, netuid, wallet, uids):
        retries = 5
        backoff = 1.5
        msg = None
        success = False
        for attempt in range(retries):
            try:
                success, msg = self.subtensor.set_weights(
                    netuid=netuid,
                    wallet=wallet,
                    uids=uids,
                    weights=weights,
                    wait_for_inclusion=False,
                    wait_for_finalization=False,
                    version_key=weights_version_key,
                )
                if success:
                    return True
            except Exception as e:
                if attempt == retries - 1:
                    raise e
                wait_time = backoff**attempt

                bt.logging.error(
                    f"Failed to set weights {msg} (attempt {attempt+1}/{retries}). Retrying in {wait_time:.1f}s..."
                )
                self.close_subtensor()
                self._reinit_subtensor()
                time.sleep(wait_time)
        return False

    async def _try_set_weights(self) -> Tuple[bool, Optional[str]]:
        weights_success = False
        error_str = None
        try:
            cpu_weights = self.weights
            adjusted_weights = cpu_weights

            self.weights.nan_to_num(0.0)
            try:
                weights_success = await asyncio.wait_for(
                    self.set_weights_with_wait(
                        weights=adjusted_weights,
                        netuid=self.config.netuid,
                        wallet=self.wallet,
                        uids=self.metagraph.uids,
                    ),
                    timeout=600,  # 10 minutes
                )
            except asyncio.TimeoutError:
                bt.logging.error("Setting weights timed out after 10 minutes")
                weights_success = False
            weights_report = {"weights": {}}
            for uid, score in enumerate(self.weights):
                weights_report["weights"][uid] = score
            bt.logging.debug(f"set_weights_complete: {weights_report}")
            bt.logging.warning(f"successfully_set_weights")
            weights_success = True
        except Exception as e:
            bt.logging.error(f"failed_set_weights error={e}\n{traceback.format_exc()}")
            error_str = f"failed_set_weights error={e}\n{traceback.format_exc()}"

            return weights_success, error_str

        # Only dump weight state to console
        ws, ui = self.weights.topk(len(self.weights))
        table = Table(title="All Weights")
        table.add_column("uid", justify="right", style="cyan", no_wrap=True)
        table.add_column("weight", style="magenta")
        for index, weight in list(zip(ui.tolist(), ws.tolist())):
            table.add_row(str(index), str(round(weight, 4)))
        console = Console()
        console.print(table)

        # Weight setting status
        status_table = Table(title="Weight Setting Status")
        status_table.add_column("Status", style="cyan")
        status_table.add_column("Value", style="magenta")
        status_table.add_row("successfully_set_weights", str(weights_success))
        weights_failed = not weights_success
        status_table.add_row("failed_set_weights", str(weights_failed))
        console.print(status_table)
        return weights_success, error_str

    async def try_set_weights(self, ttl: int) -> Tuple[bool, Optional[str]]:
        if self.config.offline:
            return False, None

        weights_set_success = False
        error_msg = None
        exception_msg = None
        try:
            bt.logging.debug("Setting weights.")
            weights_set_success, error_msg = await asyncio.wait_for(self._try_set_weights(), ttl)
            bt.logging.debug("Finished setting weights.")
        except asyncio.TimeoutError:
            error_msg = f"Failed to set weights after {ttl} seconds"
            bt.logging.error(error_msg)
        except Exception as e:
            exception_msg = f"Error setting weights: {e}\n{traceback.format_exc()}"
            bt.logging.error(exception_msg)
        finally:
            payload = {
                "time": str(dt.datetime.now(dt.timezone.utc)),
                "weights_set_success": weights_set_success,
                "error": error_msg,
                "exception_msg": exception_msg,
                "weights_version": weights_version_key,
            }
            self._remote_log(payload)
        return weights_set_success, error_msg

    @staticmethod
    def get_miner_scores(
        hotkeys: List[str],
        config,
        local_metadata: LocalMetadata,
        signatures: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Get scores for multiple miners.

        Args:
            hotkeys: List of hotkeys to get scores for
            config: Validator config
            local_metadata: Local metadata for the validator
            signatures: Request signatures
            debug: Debug flag

        Returns:
            Dict with structure:
            {
                "scores": {
                    "hotkey1": {"status": "VALID", "total_score": 0.5},
                    "hotkey2": {"status": "INVALID", "total_score": 0.0},
                    ...
                },
                "stats": {}
            }
        """

        # Construct URL with query parameters
        validation_endpoint = f"{ORCHESTRATION_SERVER}/scores"

        # Prepare request data
        request_data = {
            "hotkeys": hotkeys
        }

        # Set up headers
        headers = {
            "Git-Commit": str(local_metadata.commit),
            "Bittensor-Version": str(local_metadata.btversion),
            "UID": str(local_metadata.uid),
            "Hotkey": str(local_metadata.hotkey),
            "Coldkey": str(local_metadata.coldkey),
        }
        headers.update(signatures)

        # Initialize response structure
        response_data = {
            "scores": {},
            "stats": {}
        }

        try:
            # Make POST request with list of hotkeys
            response = requests.get(validation_endpoint, json=request_data, headers=headers)
            response.raise_for_status()

            # Parse response
            result = response.json()

            if result and "scores" in result:
                # Process each hotkey's score data
                for hotkey in hotkeys:
                    if hotkey in result["scores"]:
                        score_info = result["scores"][hotkey]

                        # Map old status values to new ones
                        old_status = score_info.get("status", "FAILED")
                        if old_status in ["COMPLETED"]:
                            new_status = "VALID"
                        elif old_status in ["FAILED", "QUEUED", "PRECHECK", "RUNNING"]:
                            new_status = "INVALID"
                        else:
                            new_status = old_status  # Keep new status values as-is

                        # Calculate total score
                        total_score = 0.0
                        if "score" in score_info and old_status == "COMPLETED":
                            # Use total_score directly from the score info
                            total_score = score_info["score"].get("total_score", 0.0)

                        response_data["scores"][hotkey] = {
                            "status": new_status,
                            "total_score": total_score
                        }
                    else:
                        # Hotkey not found in response
                        response_data["scores"][hotkey] = {
                            "status": "INVALID",
                            "total_score": 0.0
                        }

                # Add stats if present
                if "stats" in result:
                    response_data["stats"] = result["stats"]
            else:
                # No scores in response, mark all as invalid
                for hotkey in hotkeys:
                    response_data["scores"][hotkey] = {
                        "status": "INVALID",
                        "total_score": 0.0
                    }

        except Exception as e:
            bt.logging.error(f"Failed to get scores for hotkeys: {e}")
            # Mark all hotkeys as invalid on error
            for hotkey in hotkeys:
                response_data["scores"][hotkey] = {
                    "status": "INVALID",
                    "total_score": 0.0
                }

        bt.logging.debug(f"Retrieved scores for {len(hotkeys)} hotkeys")
        return response_data

    async def build_registry(
        self, all_uids: List[int], current_block: int
    ) -> Tuple[List[int], Dict[int, MinerEntry]]:
        miner_registry: Dict[int, MinerEntry] = {uid: MinerEntry() for uid in all_uids}
        invalid_uids = []

        # Collect all hotkeys
        hotkeys = []
        uid_to_hotkey = {}
        for uid in all_uids:
            hotkey = self.metagraph.hotkeys[uid]
            hotkeys.append(hotkey)
            uid_to_hotkey[uid] = hotkey
            miner_registry[uid].hotkey = hotkey

        bt.logging.debug(f"Getting scores for {len(hotkeys)} hotkeys")

        # Generate signature for the request
        request_signature = sign_request(
            keypair=self.wallet.hotkey,
            payload=self.local_metadata.hotkey
        )

        # Get all scores in batch
        hotkey_scores = Validator.get_miner_scores(
            hotkeys=hotkeys,
            config=self.config,
            local_metadata=self.local_metadata,
            signatures=request_signature
        )

        # Update miner registry with scores
        for uid in all_uids:
            hotkey = uid_to_hotkey[uid]

            try:
                if hotkey in hotkey_scores["scores"]:
                    score_data = hotkey_scores["scores"][hotkey]

                    if score_data["status"] == "VALID":
                        miner_registry[uid].total_score = score_data["total_score"]
                        bt.logging.warning(
                            f"uid={uid} status_complete on block {miner_registry[uid].block} : {miner_registry[uid].hotkey} {score_data}"
                        )
                    elif score_data["status"] == "INVALID":
                        bt.logging.warning(
                            f"uid={uid} status_failed on block {miner_registry[uid].block} : {miner_registry[uid].hotkey}"
                        )
                        miner_registry[uid].total_score = 0
                else:
                    bt.logging.error(f"No score data found for uid={uid} hotkey={hotkey}")
                    invalid_uids.append(uid)

            except Exception as e:
                bt.logging.error(f"Could not update for uid={uid}:{hotkey} {e}")
                bt.logging.error(f"Traceback: {traceback.format_exc()}")
                invalid_uids.append(uid)

        return invalid_uids, miner_registry

    @staticmethod
    def new_subtensor(config=None):
        if config is not None:
            # Check if network is explicitly specified
            if hasattr(config, 'network') and config.network:
                if config.network == "local":
                    subtensor = Subtensor(config=config)
                else:
                    subtensor = Subtensor(network=config.network)
                bt.logging.warning(f"subtensor initialized with network: {config.network}")
                return subtensor
            # Fall back to local flag
            elif hasattr(config, 'local') and config.local:
                subtensor = Subtensor(config=config)
                return subtensor
            else:
                subtensor = Subtensor(config=config)
                return subtensor
        subtensor = Subtensor(network="finney")
        bt.logging.warning(f"subtensor retry initialized with Subtensor(): {subtensor}")
        return subtensor

    def close_subtensor(self):
        status = ""
        try:
            self.subtensor.close()
            status = "subtensor_closed"
        except Exception as e:
            status = f"{str(e)}\n{traceback.format_exc()}"
            self.subtensor = None
        payload = {"subtensor_close_status": status}
        self._remote_log(payload)

    async def try_sync_metagraph(self, ttl: int = 120) -> bool:
        # Determine network based on config
        if hasattr(self.config, 'network') and self.config.network:
            network = self.config.network
        elif self.config.local:
            network = "local"
        else:
            network = "finney"
        try:
            bt.logging.warning(f"attempting sync with network {network}")

            self.metagraph = Metagraph(netuid=self.config.netuid, network=network, lite=False, sync=True)
            return True
        except Exception as e:
            metagraph_failure_payload = {
                "initial_metagraph_sync_success": False,
                "failure_str": str(e),
                "stacktrace": traceback.format_exc(),
                "network": network,
            }
            self._remote_log(metagraph_failure_payload)
            bt.logging.error(
                f"could not sync metagraph {e} using network {network}. Starting retries. If this issue persists please restart the valdiator script"
            )
            self.close_subtensor()
            self._reinit_subtensor()

        def sync_metagraph(attempt):
            try:
                self.metagraph.sync(block=None, lite=False, subtensor=self.subtensor)
            except Exception as e:
                bt.logging.error(f"{e}")
                # Log failure to sync metagraph
                metagraph_failure_payload = {
                    "metagraph_sync_success": False,
                    "failure_str": str(e),
                    "attempt": attempt,
                    "stacktrace": traceback.format_exc(),
                }
                self._remote_log(metagraph_failure_payload)
                self.close_subtensor()
                self._reinit_subtensor()
                raise e

        for attempt in range(3):
            try:
                sync_metagraph(attempt)
                return True
            # catch isues with crafting new subtensor
            except Exception as e:
                bt.logging.error(f"could not sync metagraph {e}")
                if attempt == 2:
                    return False

        bt.logging.success("Synced metagraph")
        bt.logging.debug("metagraph_sync_success")
        return True

    async def try_run_step(self, ttl: int) -> bool:
        async def _try_run_step():
            success = await self.run_step()
            return success

        try:
            bt.logging.warning(f"Running step with ttl {ttl}")
            step_success = await asyncio.wait_for(_try_run_step(), ttl)
            bt.logging.warning("Finished running step.")

            return step_success
        except asyncio.TimeoutError:
            bt.logging.error(f"Failed to run step after {ttl} seconds")
            return False
        except Exception as e:
            bt.logging.error(f"Failed to run step : {e} {traceback.format_exc()}")
            return False

    async def run_step(self) -> bool:
        """
        Executes a step in the evaluation process of models. This function performs several key tasks:
        1. Iterate through blockchain state to find miner entries for models.
        2. Fetches model scoring data from separate evaluation instance.
        3. Applies elimination logic to better calulate model scoring.
        4. Calculates wins and win rates for each model to determine their performance relative to others.
        5. Updates the weights of each model based on their performance and applies a softmax normalization.
        6. Logs all relevant data for the step, including model IDs, scores, and win rates.
        """

        # Update self.metagraph
        synced = await self.try_sync_metagraph(ttl=120)
        if not synced:
            bt.logging.error("could not sync metagraph. skipping this epoch")
            return False
        current_block = self.metagraph.block.item()
        try:
            payload = {"current_block": current_block, "metagraph_version": str(self.metagraph.version)}
            Validator.telemetry_report(local_metadata=self.local_metadata, payload=payload)
        except Exception as e:
            bt.logging.error(f"could not report telemetry {e}")

        all_uids = self.metagraph.uids.tolist()
        # Avoid biasing lower value uids when making calls
        random.shuffle(all_uids)
        # Prepare evaluation
        bt.logging.debug(
            f"Computing metrics on {len(all_uids)} uids"
        )

        invalid_uids, miner_registry = await self.build_registry(all_uids=all_uids, current_block=current_block)
        bt.logging.warning(f"invalid_uids : {invalid_uids}")

        bt.logging.warning(
            f"all_uids : {len(miner_registry)} invalid uids: {len(invalid_uids)} cutoff_block : {current_block}"
        )
        # Mark uids that do not have a proper score
        for uid in invalid_uids:
            if uid not in miner_registry:
                miner_registry[uid] = MinerEntry()
            miner_registry[uid].invalid = True
            miner_registry[uid].total_score = 0

        # Compute wins and win rates per uid.
        wins, win_rate = Validator.score_to_winrate(miner_registry)
        sorted_uids = sorted(miner_registry.keys())

        # Compute weights based on win rate
        model_weights = torch.tensor([win_rate[uid] for uid in sorted_uids], dtype=torch.float32)

        target_uid = SUBNET_OWNER_UID

        burn_rate = self.calculate_dynamic_burn_rate(current_block)
        bt.logging.info(f"Using fixed burn rate: {burn_rate:.6f}")

        # Special case: all scores are zero -> allocate all weight to SUBNET_OWNER_UID if present
        scores_all_zero = all(miner_registry[uid].total_score == 0 for uid in sorted_uids)
        if scores_all_zero:
            step_weights = torch.zeros(len(sorted_uids), dtype=torch.float32)
            try:
                target_idx = sorted_uids.index(target_uid)
                step_weights[target_idx] = 1.0
                bt.logging.warning(
                    f"All scores zero; allocating all weight to owner uid {target_uid}"
                )
            except ValueError:
                # Target uid not in current metagraph; fall back to uniform across available uids
                if len(sorted_uids) > 0:
                    step_weights.fill_(1.0 / len(sorted_uids))
                    bt.logging.warning(
                        f"All scores zero; owner uid {target_uid} missing; using uniform weights"
                    )
        else:
            temp = temperature
            base_weights = torch.softmax(model_weights / temp, dim=0)

            step_weights = self.adjust_weights_for_burn(
                base_weights=base_weights,
                sorted_uids=sorted_uids,
                burn_rate=burn_rate,
                target_uid=target_uid
            )


        # Safely normalize step weights (no-op if empty)
        if step_weights.numel() > 0:
            denom = step_weights.sum()
            if denom > 0:
                step_weights = step_weights / denom

        # Update weights based on moving average.
        torch_metagraph = torch.from_numpy(self.metagraph.S).to(torch.float32)
        self.weights = torch.zeros_like(torch_metagraph, dtype=torch.float32)
        new_weights = torch.zeros_like(torch_metagraph, dtype=torch.float32)
        for i, uid_i in enumerate(sorted_uids):
            new_weights[uid_i] = step_weights[i]
        # Normalize new_weights safely (avoid in-place float->long cast and zero division)
        total = new_weights.sum()
        if total > 0:
            new_weights = new_weights / total
        if new_weights.shape[0] < self.weights.shape[0]:
            self.weights = self.weights[: new_weights.shape[0]]
        elif new_weights.shape[0] > self.weights.shape[0]:
            self.weights = torch.cat(
                [
                    self.weights,
                    torch.zeros(new_weights.shape[0] - self.weights.shape[0]),
                ]
            )
        self.weights = alpha * self.weights + (1 - alpha) * new_weights
        self.weights = self.weights.nan_to_num(0.0)

        # Log to screen.
        self.log_step(
            miner_registry,
            wins,
            win_rate,
            current_block,
            burn_rate,
        )

        return True

    def log_step(
        self,
        miner_registry: Dict[int, MinerEntry],
        wins,
        win_rate,
        current_block: int,
        burn_rate: float,
    ):
        sorted_uids = sorted(miner_registry.keys())
        # Build step log
        step_log = {
            "timestamp": time.time(),
            "uids": sorted_uids,
            "uid_data": {},
            "current_block": current_block,
            "burn_rate": burn_rate,
        }
        for i, uid in enumerate(sorted_uids):
            step_log["uid_data"][str(uid)] = {
                "uid": uid,
                "block": miner_registry[uid].block,
                "score": miner_registry[uid].total_score,
                "win_rate": win_rate[uid],
                "win_total": wins[uid],
                "weight": self.weights[uid].item(),
            }

        burn_table = Table(title="Fixed Burn Information")
        burn_table.add_column("Metric", style="cyan")
        burn_table.add_column("Value", style="magenta")
        burn_table.add_row("Current Block", str(current_block))
        burn_table.add_row("Fixed Burn Rate", f"{burn_rate:.6f}")
        burn_table.add_row("Target UID (Owner)", str(SUBNET_OWNER_UID))
        console = Console()
        console.print(burn_table)

        table = Table(title="Step")
        table.add_column("uid", justify="right", style="cyan", no_wrap=True)
        table.add_column("score", style="magenta")
        table.add_column("win_rate", style="magenta")
        table.add_column("win_total", style="magenta")
        table.add_column("weights", style="magenta")
        table.add_column("block", style="magenta")
        for uid in sorted_uids:
            try:
                table.add_row(
                    str(uid),
                    str(round(step_log["uid_data"][str(uid)]["score"], 4)),
                    str(round(step_log["uid_data"][str(uid)]["win_rate"], 4)),
                    str(step_log["uid_data"][str(uid)]["win_total"]),
                    str(round(self.weights[uid].item(), 4)),
                    str(step_log["uid_data"][str(uid)]["block"]),
                )
            except:
                pass
        console = Console()
        console.print(table)

        ws, ui = self.weights.topk(len(self.weights))
        table = Table(title="Weights > 0.001")
        table.add_column("uid", justify="right", style="cyan", no_wrap=True)
        table.add_column("weight", style="magenta")
        for index, weight in list(zip(ui.tolist(), ws.tolist())):
            if weight > 0.001:
                table.add_row(str(index), str(round(weight, 4)))
        console = Console()
        console.print(table)

        # Sink step log.
        bt.logging.debug(f"Step results: {step_log}")
        scores_per_uid = {}
        for uid in sorted_uids:
            scores_per_uid[uid] = miner_registry[uid].total_score
        bt.logging.debug(f"log_scores: {scores_per_uid}")

        bt.logging.info(
            f"Fixed burn summary - Block: {current_block}, Rate: {burn_rate:.6f}"
        )

    async def run(self):
        while True:
            try:
                current_time = dt.datetime.now(dt.timezone.utc)
                minutes = current_time.minute

                # Check if we're at a 20-minute mark (with 2 minute leeway)
                if minutes % 20 <= 2 or self.config.immediate:
                    bt.logging.success(f"Running step at {current_time.strftime('%H:%M')}")
                    run_step_success = False
                    final_run_step_attempt = 0
                    for attempt in range(3):
                        final_run_step_attempt = attempt
                        try:
                            # Allow run step to execute for 30m
                            run_step_success = await self.try_run_step(ttl=60 * 30)
                            if run_step_success:
                                break
                            # If not successful, but no exception, it implies a handled failure within try_run_step
                            # Log this attempt's failure and retry if not the last attempt
                            bt.logging.warning(f"try_run_step attempt {attempt + 1} failed. Retrying...")

                        except Exception as e:
                            bt.logging.error(f"Exception during try_run_step attempt {attempt + 1}: {e}")
                            run_step_success = False # Ensure success is false on exception

                        if not run_step_success and attempt < 2: # If failed and not the last attempt
                            wait_time = (2**attempt) * 5  # 5s, 10s, 20s backoff
                            bt.logging.warning(f"Retrying try_run_step in {wait_time}s...")
                            await asyncio.sleep(wait_time)
                        elif not run_step_success and attempt == 2:
                             bt.logging.error(f"Failed all 3 attempts to run step.")

                    run_step_payload = {
                        "run_step_success": run_step_success,
                        "attempts_made": final_run_step_attempt + 1
                    }
                    self._remote_log(run_step_payload)

                    weights_set_success = False
                    error_msg = None
                    if run_step_success:
                        pre_weights_payload = {
                            "step": "pre_weights",
                            "subtensor": str(self.subtensor),
                        }
                        self._remote_log(pre_weights_payload)
                        try:
                            weights_set_success, error_msg = await self.try_set_weights(ttl=120)
                            bt.logging.warning(f"weights_set_success {weights_set_success} error_msg {error_msg}")
                        except Exception as e:
                            bt.logging.error(f"Error setting weights: {e}\n{traceback.format_exc()}")
                            weights_set_success = False
                            error_msg = f"{error_msg or ''}\nException: {e}\n{traceback.format_exc()}"

                    # Construct the payload directly for _remote_log
                    final_status_payload = {
                        "run_step_success": run_step_success,
                        "try_weights_complete": weights_set_success,
                        "error_msg": error_msg,
                    }
                    self._remote_log(final_status_payload)

                    if self.config.immediate:
                        if weights_set_success and run_step_success:
                            print("success run. exiting now")
                            os._exit(0)
                            return
                        else:
                            print("failed run. exiting now")
                            os._exit(1)
                            return
                        
                    # Wait for ~1 minute to avoid running multiple times within the same minute
                    await asyncio.sleep(70)

                else:
                    current_time = dt.datetime.now(dt.timezone.utc)
                    if current_time.minute % 5 == 0:
                        metagraph_sync_success = await self.try_sync_metagraph(ttl=300)
                        if not metagraph_sync_success:
                            try:
                                self._reinit_subtensor()
                            except Exception as e:
                                bt.logging.error(f"Error in initializing subtensor:  {e} \n {traceback.format_exc()}")

                    current_time = dt.datetime.now(dt.timezone.utc)
                    minutes = current_time.minute
                    minutes_until_next = 20 - (minutes % 20)
                    next_run = current_time + dt.timedelta(minutes=minutes_until_next)
                    bt.logging.warning(
                        f"Waiting {minutes_until_next} minutes until next run at {next_run.strftime('%H:%M')}"
                    )

                    # Wait until the next minute before checking again
                    await asyncio.sleep(minutes_until_next)

            except KeyboardInterrupt:
                bt.logging.warning("KeyboardInterrupt caught")
                exit()
            except Exception as e:
                bt.logging.error(f"Error in validator loop \n {e} \n {traceback.format_exc()}")
                # Construct the payload directly for _remote_log
                error_payload = {
                    "validator_loop_error": str(e),
                    "stacktrace": traceback.format_exc(),
                }
                self._remote_log(error_payload)
                # Add a small delay before retrying in case of continuous errors
                await asyncio.sleep(5)

    @staticmethod
    def telemetry_report(local_metadata: LocalMetadata, payload=None):
        telemetry_endpoint = f"{VALIDATION_SERVER}/telemetry_report"

        headers = {
            "Git-Commit": str(local_metadata.commit),
            "Bittensor-Version": str(local_metadata.btversion),
            "UID": str(local_metadata.uid),
            "Hotkey": str(local_metadata.hotkey),
            "Coldkey": str(local_metadata.coldkey),
        }

        if payload is None:
            payload = {"empty": True}
        # Make the POST request to the validation endpoint
        try:
            response = requests.post(telemetry_endpoint, json=payload, headers=headers)
            response.raise_for_status()  # Raise an exception for HTTP errors
        except Exception as e:
            bt.logging.error(e)
        return


def sign_request(keypair, payload: str):
    signed_payload = keypair.sign(data=payload)
    signed_payload_base64 = base64.b64encode(signed_payload).decode("utf-8")

    return {
        "payload_signed": signed_payload_base64,
        "payload": payload,
    }


if __name__ == "__main__":
    metadata = local_metadata()
    asyncio.run(Validator(metadata).run())
