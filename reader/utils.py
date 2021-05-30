from functools import wraps
from time import process_time
import json

def measure(func):
    @wraps(func)
    def _time_it(*args, **kwargs):
        start = int(round(process_time() * 1000))
        try:
            return func(*args, **kwargs)
        finally:
            end_ = int(round(process_time() * 1000)) - start
            print(
                f"Total execution time {func.__name__}: {end_ if end_ > 0 else 0} ms"
            )

    return _time_it

def jsonify(inference_output):
        json_result = json.dumps(inference_output , indent = 4)
        return [json_result]