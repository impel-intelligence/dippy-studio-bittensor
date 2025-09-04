"""
Periodic validator runner with auto-update.

What this does:
- Before every run: checks git for updates and upgrades packages when updated.
- Runs the validator once with "--immediate", then sleeps 20 minutes.
- Repeats forever.

Notes:
- Command-line args after this script are forwarded directly to
  `validator/validator.py`, e.g. wallet params:
    python3 scripts/start_validator.py --wallet.name my-wallet --wallet.hotkey my-hotkey
- This uses the same Python interpreter/venv as used to invoke this script.
- No external local constants are imported; required constants are declared inline.
"""

import argparse
import logging
import subprocess
import sys
import time
from datetime import timedelta
from shlex import split
from typing import List
import datetime
from pathlib import Path

log = logging.getLogger(__name__)

# ---------------------------------
# Inline constants (no local imports)
# ---------------------------------
class constants:
    # Root directory of the project (repo root)
    ROOT_DIR = Path(__file__).resolve().parent.parent
    # Remote server used for optional telemetry/remote logs
    VALIDATION_SERVER = "https://sn11.dippy-bittensor-subnet.com"

# How long to sleep between runs
SLEEP_TIME = timedelta(minutes=20)
UPDATES_CHECK_TIME = timedelta(minutes=1)

# Validator entry point path, relative to repo root
VALIDATOR_ENTRY = constants.ROOT_DIR / "validator" / "validator.py"


def get_version() -> str:
    """Extract the version as current git commit hash"""
    result = subprocess.run(
        split("git rev-parse HEAD"),
        check=True,
        capture_output=True,
        cwd=constants.ROOT_DIR,
    )
    commit = result.stdout.decode().strip()
    assert len(commit) == 40, f"Invalid commit hash: {commit}"
    return commit[:8]


def run_validator_once(args: List[str]) -> int:
    """Run the validator once with --immediate and return the exit code."""
    assert sys.executable, "Failed to get python executable"
    if not VALIDATOR_ENTRY.exists():
        raise FileNotFoundError(f"Validator entry not found: {VALIDATOR_ENTRY}")

    # Ensure --immediate is present (deduplicate if already provided)
    forwarded = [a for a in args if a != "--immediate"] + ["--immediate"]

    cmd = [
        sys.executable,
        str(VALIDATOR_ENTRY),
        *forwarded,
    ]
    log.info("Running validator once: %s", " ".join(map(str, cmd)))
    proc = subprocess.run(cmd, cwd=constants.ROOT_DIR)
    return proc.returncode


import requests
from typing import Dict, Any


def _remote_log(payload: Dict[str, Any]):

    final_payload = {
        "signature": "x",
        "payload": payload,
        "commit": "x",
        "btversion": "x",
        "uid": "0",
        "hotkey": "x",
        "coldkey": "x",
    }
    event_report_endpoint = f"{constants.VALIDATION_SERVER}/event_report"
    try:
        response = requests.post(event_report_endpoint, json=final_payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        log.info(f"successfully sent event_report with payload {final_payload}")
    except Exception as e:
        log.error(f"could not remote log: {e}. This error is ok to ignore if you are a validator")


def stop_validator_process(process: subprocess.Popen) -> None:
    # No-op: PM2 management removed for periodic single-run behavior
    return None


def pull_latest_version() -> None:
    """
    Pull the latest version from git.
    This uses `git pull --rebase`, so if any changes were made to the local repository,
    this will try to apply them on top of origin's changes. This is intentional, as we
    don't want to overwrite any local changes. However, if there are any conflicts,
    this will abort the rebase and return to the original state.
    The conflicts are expected to happen rarely since validator is expected
    to be used as-is.
    """
    try:
        subprocess.run(split("git pull --rebase --autostash"), check=True, cwd=constants.ROOT_DIR)
    except subprocess.CalledProcessError as exc:
        log.error("Failed to pull, reverting: %s", exc)
        _remote_log({"error": str(exc), "message": "Failed to pull from git, reverting"})

        subprocess.run(split("git rebase --abort"), check=True, cwd=constants.ROOT_DIR)


def upgrade_packages() -> None:
    """
    Upgrade python packages by running `pip install --upgrade -r requirements.txt`.
    Notice: this won't work if some package in `requirements.txt` is downgraded.
    Ignored as this is unlikely to happen.
    """
    log.info("Upgrading requirements")
    try:
        # Prefer validator-specific requirements if present; fallback to generic requirements.txt
        req_file = None
        for candidate in ("requirements.validator.txt", "requirements.txt"):
            if (constants.ROOT_DIR / candidate).exists():
                req_file = candidate
                break
        if req_file:
            subprocess.run(
                split(f"{sys.executable} -m pip install -r {req_file}"),
                check=True,
                cwd=constants.ROOT_DIR,
            )
        else:
            log.info("No requirements file found; skipping -r install step")
    except subprocess.CalledProcessError as exc:
        log.error("Failed to upgrade packages from requirements, proceeding anyway. %s", exc)

    log.info("Upgrading packages")
    try:
        subprocess.run(
            split(f"{sys.executable} -m pip install -e ."),
            check=True,
            cwd=constants.ROOT_DIR,
        )
    except subprocess.CalledProcessError as exc:
        log.error("Failed to upgrade packages, proceeding anyway. %s", exc)


def main(args: List[str]) -> None:
    """Run validator once, sleep, and repeat forever with auto-update before each run."""
    current_version = get_version()
    log.info("Starting periodic validator. Current version: %s", current_version)
    
    log.info(f"Starting periodic validator. Current args: {args}")


    while True:
        try:
            # 1) Check for updates (git)
            pull_latest_version()
            latest_version = get_version()
            log.info("Latest version: %s", latest_version)
            _remote_log(
                {
                    "current_version": str(current_version),
                    "latest_version": str(latest_version),
                    "message": "periodic_validator_check_update",
                }
            )

            # 2) If updated, upgrade packages
            if latest_version != current_version:
                log.info("Repository updated: %s -> %s. Upgrading packages...", current_version, latest_version)
                upgrade_packages()
                current_version = get_version()
                payload = {
                    "current_version": str(current_version),
                    "latest_version": str(latest_version),
                    "time": str(datetime.datetime.now(datetime.timezone.utc)),
                }
                _remote_log(payload)

            # 3) Run the validator once with --immediate
            rc = run_validator_once(args)
            log.info("Validator run completed with exit code %s", rc)

        except KeyboardInterrupt:
            log.info("Interrupted by user. Exiting.")
            return
        except Exception as e:
            log.error("Error during periodic run: %s", e)
            _remote_log({"error": str(e), "message": "periodic_validator_error"})

        # 4) Sleep between runs
        log.info("Sleeping for %s minutes before next run...", int(SLEEP_TIME.total_seconds() // 60))
        time.sleep(SLEEP_TIME.total_seconds())


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(
        description="Periodically run validator with auto-update; forwards extra args to validator.",
        epilog=(
            "Example: python scripts/start_validator.py "
            "--wallet.name my-wallet --wallet.hotkey my-hotkey --network finney"
        ),
        add_help=True,
    )
    # We don't define wallet args here; they are forwarded as-is.
    _, extra_args = parser.parse_known_args()

    main(extra_args)
