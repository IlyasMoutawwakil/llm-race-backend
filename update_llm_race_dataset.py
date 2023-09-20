import os
import warnings
import subprocess

import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from huggingface_hub import Repository

from open_llm_leaderboard_utils import get_eval_results

warnings.simplefilter(action="ignore", category=FutureWarning)
OPEN_LLM_RESULTS = "https://huggingface.co/datasets/open-llm-leaderboard/results"
OPEN_LLM_RACE = "https://huggingface.co/datasets/IlyasMoutawwakil/llm-race-dataset"
RESULTS_DIR = "open-llm-leaderboard-results"
RACE_DIR = "llm-race-dataset"

# Get all commits from the open-llm-results repo
shutil.rmtree(RESULTS_DIR, ignore_errors=True)
results_repo = Repository(
    repo_type="dataset",
    local_dir=RESULTS_DIR,
    clone_from=OPEN_LLM_RESULTS,
    token=os.environ["HF_TOKEN"],
)
os.chdir(RESULTS_DIR)
logs = subprocess.run(
    ["git", "log", "--pretty=format:'%H;%s;%an;%ad'"],
    capture_output=True,
    text=True,
)
logs = [c.strip("'") for c in logs.stdout.split("\n") if c != ""]
os.chdir("..")


race_repo = Repository(
    repo_type="dataset",
    local_dir=RACE_DIR,
    clone_from=OPEN_LLM_RACE,
    token=os.environ["HF_TOKEN"],
)

if os.path.exists(f"{RACE_DIR}/llm-race-dataset.parquet"):
    llm_race_dataset_df = pd.read_parquet(
        f"{RACE_DIR}/llm-race-dataset.parquet",
        engine="pyarrow",
    )
    previous_commits = llm_race_dataset_df["commit"].unique().tolist()
else:
    llm_race_dataset_df = pd.DataFrame()
    previous_commits = []


for log in tqdm(logs):
    commit = log.split(";")[0]

    if commit in previous_commits:
        continue

    date = log.split(";")[-1].split(" +")[0]
    print(f"Processing commit {commit} from {date}")

    try:
        results_repo.git_checkout(commit)
        commit_results = get_eval_results(RESULTS_DIR)
        commit_df = pd.DataFrame(commit_results)
        commit_df["commit"] = commit
        commit_df["date"] = pd.to_datetime(
            log.split(";")[-1].split(" +")[0], format="%a %b %d %H:%M:%S %Y"
        )
        commit_df["score"] = commit_df["results"].apply(
            lambda x: round(np.mean(list(x.values())), 1)
        )
        # concat the results with the existing dataset
        llm_race_dataset_df = pd.concat(
            [llm_race_dataset_df, commit_df], ignore_index=True
        )
    except Exception as e:
        print(e)
        continue

current_commits = llm_race_dataset_df["commit"].unique().tolist()

if previous_commits != current_commits:
    llm_race_dataset_df.to_parquet(
        f"{RACE_DIR}/llm-race-dataset.parquet",
        engine="pyarrow",
    )
    race_repo.git_add(".")
    race_repo.git_commit("update")
    race_repo.git_push()

# Delete local repos
shutil.rmtree(RESULTS_DIR, ignore_errors=True)
shutil.rmtree(RACE_DIR, ignore_errors=True)
