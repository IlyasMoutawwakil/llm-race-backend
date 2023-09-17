import os
import json
import numpy as np
import dateutil.parser
from typing import List, Tuple

METRICS = ["acc_norm", "acc_norm", "acc", "mc2"]
BENCHMARKS = ["arc:challenge", "hellaswag", "hendrycksTest", "truthfulqa:mc"]


def parse_eval_result(json_filepath: str) -> Tuple[str, list[dict]]:
    try:
        with open(json_filepath) as fp:
            data = json.load(fp)

        for mmlu_k in [
            "harness|hendrycksTest-abstract_algebra|5",
            "hendrycksTest-abstract_algebra",
        ]:
            if mmlu_k in data["versions"] and data["versions"][mmlu_k] == 0:
                return None, []  # we skip models with the wrong version

        config = data.get("config", data.get("config_general", None))
        model = config.get("model_name", None)
        if model is None:
            model = config.get("model_args", None)

        model_sha = config.get("model_sha", "")
        eval_sha = config.get("lighteval_sha", "")
        model_split = model.split("/", 1)

        if len(model_split) == 1:
            org = None
            model = model_split[0]
            result_key = f"{model}_{model_sha}_{eval_sha}"
        else:
            org = model_split[0]
            model = model_split[1]
            model = f"{org}/{model}"
            result_key = f"{org}_{model}_{model_sha}_{eval_sha}"

        eval_results = []
        for benchmark, metric in zip(BENCHMARKS, METRICS):
            accs = np.array(
                [v[metric] for k, v in data["results"].items() if benchmark in k]
            )
            if accs.size == 0:
                continue
            mean_acc = round(np.mean(accs) * 100.0, 1)
            eval_results.append(
                {
                    "model": model,
                    "revision": "main",
                    "model_sha": model_sha,
                    "results": {benchmark: mean_acc},
                }
            )
        return result_key, eval_results

    except Exception as e:
        print(e, json_filepath)
        return None, []


def get_eval_results(eval_dir: str) -> List[dict]:
    json_filepaths = []
    for root, dir, files in os.walk(eval_dir):
        # We should only have json files in model results
        if len(files) == 0 or any([not f.endswith(".json") for f in files]):
            continue

        # Sort the files by date
        # store results by precision maybe?
        try:
            files.sort(key=lambda x: dateutil.parser.parse(x.split("_", 1)[-1][:-5]))
        except dateutil.parser._parser.ParserError:
            files = [files[-1]]

        # up_to_date = files[-1]
        for file in files:
            json_filepaths.append(os.path.join(root, file))

    eval_results = {}

    for json_filepath in json_filepaths:
        result_key, results = parse_eval_result(json_filepath)
        for eval_result in results:
            if result_key in eval_results.keys():
                eval_results[result_key]["results"].update(eval_result["results"])
            else:
                eval_results[result_key] = eval_result

    eval_results = [v for v in eval_results.values()]

    return eval_results
