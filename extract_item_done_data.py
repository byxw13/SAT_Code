"""
item_done,token
"""

import json
import re
from typing import List, Dict, Any
NUM = 5000
def extract_boxed_context(answer: str) -> str:
    """
    boxed{}16boxed{}16
    boxed{}，answer
    """
    pattern = r'\\boxed\{[^{}]*(total:\{[^{}]*\}[^{}]*)*\}'

    matches = list(re.finditer(pattern, answer))

    if not matches:
        return answer

    last_match = matches[-1]

    start_pos = max(0, last_match.start() - 16)
    end_pos = min(len(answer), last_match.end() + 16)

    result = answer[start_pos:end_pos]

    return result

def process_jsonl_file():
    """
    JSONL，item_done
    """
    input_file = "r_math_Qwen3_8B_seed3407_sat.jsonl"
    output_file = "token_lenth/math500/Qwen3_8B_sat_seed3407_item_done_results.jsonl"
    stats_file = "token_lenth/math500/Qwen3_8B_sat_seed3407_token_statistics.txt"
    json_output_file = "token_lenth/math500/Qwen3_8B_sat_seed3407_item_done_results.json"

    print(f": {input_file}")

    total_final_chain_tokens = 0
    total_token_consumed = 0
    item_done_count = 0
    processed_data = []

    large_token_count = 0
    large_token_items = []

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            line_count = 0

            for line_num, line in enumerate(f, 1):
                line_count += 1
                line = line.strip()

                if not line:
                    continue

                try:
                    data = json.loads(line)

                    if data.get('event') == 'item_done':
                        item_done_count += 1

                        index = data.get('index')
                        question = data.get('question', '')
                        generated_answer = data.get('generated_answer', '')

                        extracted_answer = extract_boxed_context(generated_answer)

                        final_chain_tokens = data.get('final_chain_tokens', 0)
                        total_tokens = data.get('total_token_consumed', 0)

                        if isinstance(final_chain_tokens, (int, float)):
                            total_final_chain_tokens += final_chain_tokens
                        if isinstance(total_tokens, (int, float)):
                            total_token_consumed += total_tokens

                        if isinstance(total_tokens, (int, float)) and total_tokens >= NUM:
                            large_token_count += 1
                            large_token_items.append({
                                'index': index,
                                'total_token_consumed': total_tokens,
                                'final_chain_tokens': final_chain_tokens,
                                'question_preview': question[:100] + '...' if len(question) > 100 else question
                            })

                        processed_item = {
                            'index': index,
                            'question': question,
                            'extracted_answer': extracted_answer,
                            'final_chain_tokens': final_chain_tokens,
                            'total_token_consumed': total_tokens
                        }

                        processed_data.append(processed_item)

                        if item_done_count % 100 == 0:
                            print(f" {item_done_count} item_done...")

                except json.JSONDecodeError as e:
                    print(f":  {line_num} JSON: {e}")
                    continue
                except Exception as e:
                    print(f":  {line_num} : {e}")
                    continue

    except FileNotFoundError:
        print(f":  {input_file} ")
        return
    except Exception as e:
        print(f": : {e}")
        return

    print(f"\n!")
    print(f": {line_count}")
    print(f" {item_done_count} item_done")

    try:
        processed_data.sort(key=lambda x: x['index'] if x['index'] is not None else float('inf'))
        print("index")
    except Exception as e:
        print(f": {e}")

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in processed_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f": {output_file}")
    except Exception as e:
        print(f": {e}")
        return

    
    try:
        with open(json_output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        print(f"JSON: {json_output_file}")
    except Exception as e:
        print(f"JSON: {e}")

    print(f"\n=== Token  ===")
    print(f"final_chain_tokens : {total_final_chain_tokens:,}")
    print(f"total_token_consumed : {total_token_consumed:,}")

    if item_done_count > 0:
        avg_final_chain = total_final_chain_tokens / item_done_count
        avg_total_tokens = total_token_consumed / item_done_count
        print(f"item_donefinal_chain_tokens: {avg_final_chain:.2f}")
        print(f"item_donetotal_token_consumed: {avg_total_tokens:.2f}")

    print(f"\n=== Token (>={NUM} tokens) ===")
    print(f"Token>={NUM}: {large_token_count}")
    print(f": {(large_token_count / item_done_count * 100):.2f}%" if item_done_count > 0 else ": N/A")

    if large_token_items:
        print(f"\n--- Token>=NUM ---")
        print("Index | Total Tokens | Final Chain Tokens | Question Preview")
        print("-" * 80)

        large_token_items.sort(key=lambda x: x['total_token_consumed'], reverse=True)

        for item in large_token_items:
            print(f"{item['index']:5d} | {item['total_token_consumed']:12,d} | {item['final_chain_tokens']:18,d} | {item['question_preview']}")

        max_tokens = max(item['total_token_consumed'] for item in large_token_items)
        min_tokens = min(item['total_token_consumed'] for item in large_token_items)
        avg_large_tokens = sum(item['total_token_consumed'] for item in large_token_items) / len(large_token_items)

        print(f"\nToken:")
        print(f"Token: {max_tokens:,}")
        print(f"Token: {min_tokens:,}")
        print(f"Token: {avg_large_tokens:.2f}")

    print(f"\n=== 5 ===")
    for i, item in enumerate(processed_data[:5]):
        print(f"\n---  {i+1} ---")
        print(f"Index: {item['index']}")
        print(f"Final Chain Tokens: {item['final_chain_tokens']}")
        print(f"Total Token Consumed: {item['total_token_consumed']}")
        print(f"Question: {item['question'][:100]}..." if len(item['question']) > 100 else item['question'])
        print(f"Extracted Answer: {item['extracted_answer']}")
        print(f"Original Answer Length: {len(item.get('original_answer', ''))}")
        print(f"Extracted Answer Length: {len(item['extracted_answer'])}")

    
    try:
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("=== Token  ===\n\n")
            f.write(f"item_done: {item_done_count}\n")
            f.write(f"final_chain_tokens : {total_final_chain_tokens:,}\n")
            f.write(f"total_token_consumed : {total_token_consumed:,}\n")
            if item_done_count > 0:
                f.write(f"item_donefinal_chain_tokens: {avg_final_chain:.2f}\n")
                f.write(f"item_donetotal_token_consumed: {avg_total_tokens:.2f}\n")

            f.write(f"\n=== Token (>={NUM} tokens) ===\n")
            f.write(f"Token>=NUM: {large_token_count}\n")
            if item_done_count > 0:
                f.write(f": {(large_token_count / item_done_count * 100):.2f}%\n")

            if large_token_items:
                f.write(f"\n--- Token>=NUM ---\n")
                f.write("Index | Total Tokens | Final Chain Tokens | Question Preview\n")
                f.write("-" * 80 + "\n")

                large_token_items_sorted = sorted(large_token_items, key=lambda x: x['total_token_consumed'], reverse=True)

                for item in large_token_items_sorted:
                    f.write(f"{item['index']:5d} | {item['total_token_consumed']:12,d} | {item['final_chain_tokens']:18,d} | {item['question_preview']}\n")

                max_tokens = max(item['total_token_consumed'] for item in large_token_items)
                min_tokens = min(item['total_token_consumed'] for item in large_token_items)
                avg_large_tokens = sum(item['total_token_consumed'] for item in large_token_items) / len(large_token_items)

                f.write(f"\nToken:\n")
                f.write(f"Token: {max_tokens:,}\n")
                f.write(f"Token: {min_tokens:,}\n")
                f.write(f"Token: {avg_large_tokens:.2f}\n")

        print(f": {stats_file}")
    except Exception as e:
        print(f": {e}")

    return {
        'item_done_count': item_done_count,
        'total_final_chain_tokens': total_final_chain_tokens,
        'total_token_consumed': total_token_consumed,
        'processed_data': processed_data,
        'large_token_count': large_token_count,
        'large_token_items': large_token_items
    }

def main():
    """"""
    print("item_done...")

    result = process_jsonl_file()

    if result:
        print("\n===  ===")
        print(f" {result['item_done_count']} item_done")
        print(f"Token")
    else:
        print("")

if __name__ == "__main__":
    main()
