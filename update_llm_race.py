import os
import warnings
import subprocess

import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from huggingface_hub import Repository

from open_llm_utils import get_eval_results

warnings.simplefilter(action="ignore", category=FutureWarning)
OPEN_LLM_RESULTS = "https://huggingface.co/datasets/open-llm-leaderboard/results"
OPEN_LLM_RACE = "https://huggingface.co/datasets/IlyasMoutawwakil/open-llm-race-dataset"
RESULTS_DIR = "open-llm-leaderboard-results"
RACE_DIR = "open-llm-race-dataset"

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

updated = False
# Push new commits to llm-race repo
race_repo = Repository(
    repo_type="dataset",
    local_dir=RACE_DIR,
    clone_from=OPEN_LLM_RACE,
    token=os.environ["HF_TOKEN"],
)
for log in tqdm(logs):
    commit = log.split(";")[0]
    # skip commits that are already in the save dir
    if os.path.exists(f"{RACE_DIR}/{commit}.csv"):
        continue

    try:
        results_repo.git_checkout(commit)
        eval_results = get_eval_results(RESULTS_DIR)
        eval_results = pd.DataFrame(eval_results)
        eval_results["commit"] = commit
        eval_results["date"] = pd.to_datetime(
            log.split(";")[-1].split(" +")[0], format="%a %b %d %H:%M:%S %Y"
        )
        eval_results["score"] = eval_results["results"].apply(
            lambda x: round(np.mean(list(x.values())), 1)
        )
        # save results
        eval_results.to_csv(f"{RACE_DIR}/{commit}.csv", index=False)
        updated = True
    except Exception as e:
        print(e)
        continue

if updated:
    # concatenate all the csv files into one
    open_llm_race_dataset = pd.DataFrame()
    for csv_file in tqdm(os.listdir(RACE_DIR)):
        if ".csv" not in csv_file:
            continue
        commit_csv = pd.read_csv(f"{RACE_DIR}/{csv_file}")
        open_llm_race_dataset = pd.concat([open_llm_race_dataset, commit_csv])
    open_llm_race_dataset.to_csv(f"{RACE_DIR}/open-llm-race-dataset.csv", index=False)

    race_repo.git_add(".")
    race_repo.git_commit("update")
    race_repo.git_push()

# Delete local repos
shutil.rmtree(RESULTS_DIR, ignore_errors=True)
shutil.rmtree(RACE_DIR, ignore_errors=True)
