"""
1. generated_answer（extract_boxed_context）
2. API
"""

import json
import re
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI

API_KEY = "xxx"
BASE_URL = "xxx"
LLM_MODEL = "xxx"

MAX_WORKERS = 2
CHECKPOINT_INTERVAL = 200
PAUSE_DURATION = 70

INPUT_FILE = "r_math_Qwen3_8B_seed3407_sat.jsonl"
PROCESSED_OUTPUT_FILE = "outcome/math500/processed_Qwen3_8B_seed3407_sat.jsonl"
EVALUATION_OUTPUT_FILE = "outcome/math500/evaluation_Qwen3_8B_seed3407_sat.jsonl"
CHECKPOINT_FILE = "outcome/math500/evaluation_checkpoint_Qwen3_8B_seed3407_sat.jsonl"
TEST_JSONL_FILE = "math500_test.jsonl"  # 

try:
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    raise SystemExit(1)

def call_llm(messages, max_tokens=2048):
    """LLM API"""
    try:
        data = {
            'model': LLM_MODEL,
            'messages': messages,
            'max_tokens': max_tokens,
            'temperature': 0.5,
        }
        response = client.chat.completions.create(**data)
        return response.choices[0].message.content.strip()
    except Exception as e:
        tqdm.write(f"\n[API ERROR] Error calling LLM API: {e}")
        return None

def evaluate_with_llm(generated_answer, standard_answer):
    """LLM"""
    if not generated_answer or not standard_answer:
        return None

    messages = [{
        "role": "user",
        "content": f"""
You are an evaluator. Your task is to determine if the generated answer matches the standard answer.
### Task:
1. Compare the generated answer with the standard answer.
2. Ignore all formatting issues and only evaluate whether the Standard Answer is contained in the Generated Answer.
3. If they are identical or logically equivalent, respond with: "The answer is correct."
4. If they do not match in terms of logical equivalence, respond with: "The answer is incorrect."
### Input:
- Generated Answer: {generated_answer}
- Standard Answer: {standard_answer}
### Output:
Provide your evaluation here:
"""
    }]

    evaluation = call_llm(messages, max_tokens=32)
    tqdm.write(f"evaluation : {evaluation}")

    if evaluation is None:
        return None

    verdict = ("The answer is correct" in evaluation or
               "the answer is correct" in evaluation or
               ("correct" in evaluation and "incorrect" not in evaluation))
    return bool(verdict)

def extract_boxed_context(answer: str) -> str:
    """
    boxed{}16boxed{}16
    boxed{}，1500
    """
    pattern = r'\\boxed\{[^{}]*(total:\{[^{}]*\}[^{}]*)*\}'

    matches = list(re.finditer(pattern, answer))

    if not matches:
        return answer[-500:]

    last_match = matches[-1]

    start_pos = max(0, last_match.start() - 16)
    end_pos = min(len(answer), last_match.end() + 16)

    result = answer[start_pos:end_pos]

    return result

def load_jsonl_file(file_path):
    """JSONL"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f":  {file_path}  {line_num} JSON: {e}")
                        continue
    except FileNotFoundError:
        print(f":  {file_path} ")
        return []
    except Exception as e:
        print(f":  {file_path} : {e}")
        return []

    return data

def load_test_data(file_path):
    """test.jsonl，"""
    problem_to_answer = {}

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        problem = data.get('problem', '')
                        answer = data.get('answer', '')

                        if problem and answer:
                            problem_to_answer[problem] = answer
                    except json.JSONDecodeError as e:
                        print(f": test.jsonl  {line_num} JSON: {e}")
                        continue
    except FileNotFoundError:
        print(f":  {file_path} ")
        return {}
    except Exception as e:
        print(f":  {file_path} : {e}")
        return {}

    print(f" {len(problem_to_answer)} ")
    return problem_to_answer

def save_jsonl_file(data, file_path):
    """JSONL"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f": {file_path}")
    except Exception as e:
        print(f" {file_path} : {e}")

def process_and_extract_answers():
    """："""
    print("=" * 60)
    print("：generated_answer")
    print("=" * 60)

    print("r_detector...")
    all_data = load_jsonl_file(INPUT_FILE)

    if not all_data:
        print(": ")
        return []

    input_data = [item for item in all_data if item.get('event') == 'item_done']

    print(f" {len(all_data)} ")
    print(f" event='item_done' : {len(input_data)} ")

    processed_data = []
    for i, item in enumerate(tqdm(input_data, desc="")):
        try:
            index = item.get('index', i)
            question = item.get('question', '')
            generated_answer = item.get('generated_answer', '')
            branch_count = item.get('branch_count', 0)

            processed_answer = extract_boxed_context(generated_answer)

            processed_item = {
                'index': index,
                'question': question,
                'original_answer': generated_answer,
                'processed_answer': processed_answer,
                'branch_count': branch_count,
                'final_chain_R': item.get('final_chain_R', 0),
                'tau_used': item.get('tau_used', 0),
                'predicted_correct': item.get('predicted_correct', False)
            }
            processed_data.append(processed_item)

        except Exception as e:
            print(f" (index: {item.get('index', i)}): {e}")
            continue

    try:
        processed_data.sort(key=lambda x: x['index'] if x['index'] is not None else float('inf'))
    except Exception as e:
        print(f": {e}")

    save_jsonl_file(processed_data, PROCESSED_OUTPUT_FILE)

    print(f"\n！ {len(processed_data)} ")
    print(f": {PROCESSED_OUTPUT_FILE}")

    return processed_data

def load_checkpoint():
    """"""
    if not os.path.exists(CHECKPOINT_FILE):
        return set()

    try:
        processed_indices = set()
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        processed_indices.add(data.get('index'))
                    except json.JSONDecodeError:
                        continue
        return processed_indices
    except Exception as e:
        print(f": : {e}")
        return set()

def save_checkpoint(result):
    """"""
    try:
        with open(CHECKPOINT_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f": : {e}")

def process_single_item(item, index, problem_to_answer):
    """"""
    try:
        question = item.get('question', '')
        processed_answer = item.get('processed_answer', '')

        standard_answer = problem_to_answer.get(question, '')

        if not standard_answer:
            print(f":  (index: {index})")
            return None

        if 'is_correct' in item and item['is_correct'] is not None:
            print(f" (index: {index})")
            return item

        is_correct = evaluate_with_llm(processed_answer, standard_answer)

        result = {
            'index': index,
            'question': question,
            'processed_answer': processed_answer,
            'standard_answer': standard_answer,
            'branch_count': item.get('branch_count', 0),
            'final_chain_R': item.get('final_chain_R', 0),
            'tau_used': item.get('tau_used', 0),
            'predicted_correct': item.get('predicted_correct', False),
            'is_correct': is_correct
        }

        return result

    except Exception as e:
        print(f" (index: {index}): {e}")
        return None

def evaluate_answers(processed_data):
    """："""
    print("\n" + "=" * 60)
    print("：")
    print("=" * 60)

    if not processed_data:
        print(": ")
        return

    print(f" {len(processed_data)} ...")

    print("...")
    problem_to_answer = load_test_data(TEST_JSONL_FILE)

    if not problem_to_answer:
        print(": ")
        return

    existing_results = {}
    if os.path.exists(EVALUATION_OUTPUT_FILE):
        print("...")
        existing_data = load_jsonl_file(EVALUATION_OUTPUT_FILE)
        for item in existing_data:
            index = item.get('index')
            if index is not None:
                existing_results[index] = item

        print(f" {len(existing_data)} ")

        need_retry_count = sum(1 for item in existing_data if item.get('is_correct') is None)
        if need_retry_count > 0:
            print(f" {need_retry_count} null，")

    print("...")
    checkpoint_indices = load_checkpoint()

    items_to_process = []
    items_already_processed = []
    items_need_retry = []

    for index, item in enumerate(processed_data):
        item_index = item.get('index')

        if item_index in existing_results:
            existing_result = existing_results[item_index]
            if existing_result.get('is_correct') is not None:
                items_already_processed.append((item, index, existing_result))
            else:
                items_need_retry.append((item, index))
                print(f" (index: {item_index}) - null")
        elif item_index not in checkpoint_indices:
            items_to_process.append((item, index))

    print(f":")
    print(f"- : {len(processed_data)}")
    print(f"- : {len(items_already_processed)}")
    print(f"- (null): {len(items_need_retry)}")
    print(f"- : {len(items_to_process)}")
    print(f"- : {len(checkpoint_indices)}")

    items_to_process.extend(items_need_retry)

    if items_already_processed and not os.path.exists(EVALUATION_OUTPUT_FILE):
        print("...")
        with open(EVALUATION_OUTPUT_FILE, 'w', encoding='utf-8') as f:
            for item, index, result in items_already_processed:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

    if not items_to_process:
        print("")

        if items_already_processed:
            correct_count = sum(1 for _, _, result in items_already_processed if result.get('is_correct') == True)
            total_count = len(items_already_processed)
            if total_count > 0:
                accuracy = correct_count / total_count * 100
                print(f": {correct_count}/{total_count} = {accuracy:.2f}%")

        return

    if not os.path.exists(EVALUATION_OUTPUT_FILE):
        with open(EVALUATION_OUTPUT_FILE, 'w', encoding='utf-8') as f:
            pass  # 

    results = []
    batch_size = CHECKPOINT_INTERVAL  # 

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        total_processed = 0

        for batch_start in range(0, len(items_to_process), batch_size):
            batch_end = min(batch_start + batch_size, len(items_to_process))
            batch_items = items_to_process[batch_start:batch_end]

            print(f"\n {batch_start//batch_size + 1}  ( {len(batch_items)} )")

            pbar = tqdm(total=len(batch_items), desc=f" ( {batch_start//batch_size + 1})")

            future_to_item = {
                executor.submit(process_single_item, item, index, problem_to_answer): (item, index)
                for item, index in batch_items
            }

            for future in as_completed(future_to_item):
                item, index = future_to_item[future]

                try:
                    result = future.result()
                    if result:
                        results.append(result)

                        with open(EVALUATION_OUTPUT_FILE, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(result, ensure_ascii=False) + '\n')

                        save_checkpoint(result)

                        total_processed += 1

                except Exception as e:
                    print(f": {e}")

                pbar.update(1)

            pbar.close()

            if batch_end < len(items_to_process):
                print(f"\n {total_processed} ， {PAUSE_DURATION} API...")
                time.sleep(PAUSE_DURATION)

    print(f"\n！")
    print(f" {len(results)} ")
    print(f": {EVALUATION_OUTPUT_FILE}")
    print(f": {CHECKPOINT_FILE}")

    if results:
        correct_count = sum(1 for r in results if r.get('is_correct') == True)
        total_count = len([r for r in results if r.get('is_correct') is not None])
        if total_count > 0:
            accuracy = correct_count / total_count * 100
            print(f": {correct_count}/{total_count} = {accuracy:.2f}%")

def analyze_existing_results():
    """"""
    print("\n" + "=" * 60)
    print("")
    print("=" * 60)

    if os.path.exists(PROCESSED_OUTPUT_FILE):
        processed_data = load_jsonl_file(PROCESSED_OUTPUT_FILE)
        if processed_data:
            print(f": {len(processed_data)}")

            with_standard_answer = [item for item in processed_data if item.get('standard_answer')]
            print(f": {len(with_standard_answer)}")

    if os.path.exists(EVALUATION_OUTPUT_FILE):
        evaluation_data = load_jsonl_file(EVALUATION_OUTPUT_FILE)
        if evaluation_data:
            print(f": {len(evaluation_data)}")

            with_correct = [item for item in evaluation_data if item.get('is_correct') is not None]
            if with_correct:
                correct_count = sum(1 for item in with_correct if item.get('is_correct') == True)
                accuracy = correct_count / len(with_correct) * 100
                print(f": {correct_count}/{len(with_correct)} = {accuracy:.2f}%")

                predicted_matches = 0
                total_predicted = 0
                for item in with_correct:
                    if 'predicted_correct' in item:
                        total_predicted += 1
                        if item['predicted_correct'] == item['is_correct']:
                            predicted_matches += 1

                if total_predicted > 0:
                    predicted_accuracy = predicted_matches / total_predicted * 100
                    print(f" (predicted_correct vs is_correct): {predicted_matches}/{total_predicted} = {predicted_accuracy:.2f}%")

if __name__ == "__main__":
    analyze_existing_results()

    processed_data = []
    if os.path.exists(PROCESSED_OUTPUT_FILE):
        print(f"\n: {PROCESSED_OUTPUT_FILE}")
        use_existing = input("？(y/n): ").lower().strip()
        if use_existing == 'y':
            processed_data = load_jsonl_file(PROCESSED_OUTPUT_FILE)
        else:
            processed_data = process_and_extract_answers()
    else:
        processed_data = process_and_extract_answers()

    if processed_data:
        evaluate_answers(processed_data)
    else:
        print(": ")
