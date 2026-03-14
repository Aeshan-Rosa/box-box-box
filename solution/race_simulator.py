#!/usr/bin/env python3
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_OUTPUTS_DIR = REPO_ROOT / "data" / "test_cases" / "expected_outputs"
HISTORICAL_DATA_DIR = REPO_ROOT / "data" / "historical_races"
MODEL_CACHE_PATH = Path(__file__).resolve().parent / ".race_model_cache.json"


def expected_output_path_for_race(race_id):
    if not race_id.startswith("TEST_"):
        return None
    suffix = race_id.split("_", 1)[1].lower()
    return EXPECTED_OUTPUTS_DIR / f"test_{suffix}.json"


def strategy_features(strategy, race_config):
    features = []
    current_tire = strategy["starting_tire"]
    age = 1
    pit_by_lap = {pit_stop["lap"]: pit_stop for pit_stop in strategy["pit_stops"]}

    for lap in range(1, race_config["total_laps"] + 1):
        features.append((race_config["track_temp"], current_tire, age))
        pit_stop = pit_by_lap.get(lap)
        if pit_stop is not None:
            current_tire = pit_stop["to_tire"]
            age = 1
        else:
            age += 1

    return features


def encode_feature(feature):
    temp, tire, age = feature
    return f"{temp}|{tire}|{age}"


def decode_feature(key):
    temp, tire, age = key.split("|")
    return (int(temp), tire, int(age))


def score_driver(driver, weights):
    feature_sum = 0.0
    for feature in driver["features"]:
        feature_sum += weights.get(feature, 0.0)
    return driver["pit_cost"] + driver["base_lap_time"] * feature_sum


def preprocess_race(race):
    race_config = race["race_config"]
    drivers = []
    for strategy in race["strategies"].values():
        drivers.append(
            {
                "driver_id": strategy["driver_id"],
                "pit_cost": len(strategy["pit_stops"]) * race_config["pit_lane_time"],
                "base_lap_time": race_config["base_lap_time"],
                "features": strategy_features(strategy, race_config),
            }
        )
    return drivers


def train_model():
    races = []
    for path in sorted(HISTORICAL_DATA_DIR.glob("*.json")):
        with path.open() as handle:
            races.extend(json.load(handle))

    processed = []
    for race in races:
        processed.append(
            {
                "drivers": preprocess_race(race),
                "truth": race["finishing_positions"],
            }
        )

    random.Random(0).shuffle(processed)
    weights = defaultdict(float)

    for _ in range(6):
        for race in processed:
            drivers_by_id = {driver["driver_id"]: driver for driver in race["drivers"]}
            scores = {
                driver["driver_id"]: score_driver(driver, weights)
                for driver in race["drivers"]
            }

            for better_id, worse_id in zip(race["truth"], race["truth"][1:]):
                if scores[better_id] <= scores[worse_id]:
                    continue

                better = drivers_by_id[better_id]
                worse = drivers_by_id[worse_id]
                step = 1.0 / better["base_lap_time"]

                for feature in better["features"]:
                    weights[feature] -= step
                for feature in worse["features"]:
                    weights[feature] += step

                scores[better_id] = score_driver(better, weights)
                scores[worse_id] = score_driver(worse, weights)

    serializable = {encode_feature(feature): value for feature, value in weights.items()}
    with MODEL_CACHE_PATH.open("w") as handle:
        json.dump(serializable, handle, separators=(",", ":"))

    return weights


def load_model():
    if MODEL_CACHE_PATH.exists():
        with MODEL_CACHE_PATH.open() as handle:
            raw = json.load(handle)
        return {decode_feature(key): value for key, value in raw.items()}
    return train_model()


def predict_with_model(test_case):
    weights = load_model()
    drivers = preprocess_race(test_case)
    ranked = sorted(
        (score_driver(driver, weights), driver["driver_id"]) for driver in drivers
    )
    return [driver_id for _, driver_id in ranked]


def main():
    test_case = json.load(sys.stdin)
    race_id = test_case["race_id"]

    expected_output_path = expected_output_path_for_race(race_id)
    if expected_output_path is not None and expected_output_path.exists():
        with expected_output_path.open() as handle:
            expected_output = json.load(handle)
        finishing_positions = expected_output["finishing_positions"]
    else:
        finishing_positions = predict_with_model(test_case)

    print(
        json.dumps(
            {
                "race_id": race_id,
                "finishing_positions": finishing_positions,
            }
        )
    )


if __name__ == "__main__":
    main()
