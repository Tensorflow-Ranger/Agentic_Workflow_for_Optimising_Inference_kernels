import json
import os
import re
import difflib

# Directory paths
sft_dataset_dir = '/Users/umamaheswari/Documents/GitHub/Agentic_Workflow_for_Optimising_Inference_kernels/sft_dataset'
generated_kernels_dir = '/Users/umamaheswari/Documents/GitHub/Agentic_Workflow_for_Optimising_Inference_kernels/generated_kernels'

# Get all JSONL files
jsonl_files = [f for f in os.listdir(sft_dataset_dir) if f.endswith('.jsonl')]

missing_kernels = []
filled_more = []

# Build list of all kernel files under generated_kernels for fuzzy matching
all_kernel_files = []  # tuples of (basename, relpath, abspath)
for root, dirs, files in os.walk(generated_kernels_dir):
    for f in files:
        rel = os.path.relpath(os.path.join(root, f), generated_kernels_dir)
        all_kernel_files.append((f, rel, os.path.join(root, f)))

def normalize_name(name: str) -> str:
    # remove common numeric tokens like _128x256, _64x64, _32x3, and align suffixes
    s = re.sub(r"_\d+x\d+", "", name)
    s = re.sub(r"_\d+x\d+_?\w*", "", s)
    s = s.replace('align', '')
    s = s.replace('__', '_')
    return s

for jsonl_file in jsonl_files:
    file_path = os.path.join(sft_dataset_dir, jsonl_file)
    print(f"Processing {jsonl_file}...")

    # Read all lines
    with open(file_path, 'r') as f:
        lines = f.readlines()

    updated_lines = []
    for i, line in enumerate(lines):
        entry = json.loads(line.strip())

        # Check if optimized_kernel is missing or empty
        if 'optimized_kernel' not in entry or not entry['optimized_kernel'].strip():
            kernel_file = entry.get('kernel_file') or entry.get('optimized_kernel_file')
            if kernel_file:
                kernel_path = None
                kernel_filename = os.path.basename(kernel_file)

                # 1) Exact path relative to repo root
                candidate = os.path.join(os.path.dirname(sft_dataset_dir), kernel_file)
                if os.path.exists(candidate):
                    kernel_path = candidate

                # 2) Direct join with generated_kernels_dir
                if kernel_path is None:
                    candidate = os.path.join(generated_kernels_dir, kernel_file)
                    if os.path.exists(candidate):
                        kernel_path = candidate

                # 3) Search by basename anywhere under generated_kernels
                if kernel_path is None:
                    for root, dirs, files in os.walk(generated_kernels_dir):
                        if kernel_filename in files:
                            kernel_path = os.path.join(root, kernel_filename)
                            break

                # 4) Fuzzy: try removing a trailing size token like '_32x3' (split on '_32x')
                if kernel_path is None and '_32x' in kernel_filename:
                    prefix = kernel_filename.split('_32x')[0]
                    for root, dirs, files in os.walk(generated_kernels_dir):
                        for f in files:
                            if f.startswith(prefix):
                                kernel_path = os.path.join(root, f)
                                break
                        if kernel_path:
                            break

                # 5) Fallback: match on a shorter prefix (first 40 chars)
                if kernel_path is None:
                    short = kernel_filename[:40]
                    for root, dirs, files in os.walk(generated_kernels_dir):
                        for f in files:
                            if short in f:
                                kernel_path = os.path.join(root, f)
                                break
                        if kernel_path:
                            break

                # 6) Aggressive fuzzy match using difflib on basenames
                if kernel_path is None:
                    basenames = [t[0] for t in all_kernel_files]
                    matches = difflib.get_close_matches(kernel_filename, basenames, n=5, cutoff=0.6)
                    if matches:
                        # choose best match that also shares significant normalized prefix
                        best = None
                        for m in matches:
                            if normalize_name(m).startswith(normalize_name(kernel_filename)[:10]):
                                best = m
                                break
                        if best is None:
                            best = matches[0]
                        # find the full path for best
                        for b, rel, a in all_kernel_files:
                            if b == best:
                                kernel_path = a
                                break

                # 7) Regex-normalized substring matching
                if kernel_path is None:
                    target_norm = normalize_name(kernel_filename)
                    for b, rel, a in all_kernel_files:
                        if target_norm and target_norm in normalize_name(b):
                            kernel_path = a
                            break

                if kernel_path:
                    print(f"  Filled missing kernel for entry {i+1}: {kernel_file} -> {os.path.relpath(kernel_path, generated_kernels_dir)}")
                    missing_kernels.append((jsonl_file, i+1, kernel_file))

                    # Read the kernel content
                    with open(kernel_path, 'r') as kf:
                        kernel_content = kf.read()

                    # Add the optimized_kernel field and remove reference
                    entry['optimized_kernel'] = kernel_content
                    if 'optimized_kernel_file' in entry:
                        del entry['optimized_kernel_file']
                else:
                    # couldn't find kernel on disk
                    print(f"  Kernel file not found for entry {i+1}: {kernel_file}")
            else:
                print(f"  No kernel_file specified for entry {i+1}")

        # Convert back to JSON
        updated_lines.append(json.dumps(entry, separators=(',', ':')))

    # Write back the file
    with open(file_path, 'w') as f:
        f.write('\n'.join(updated_lines) + '\n')

print(f"\nProcessed {len(jsonl_files)} files.")
print(f"Found and filled {len(missing_kernels)} missing kernels:")
for file, entry_num, kernel in missing_kernels:
    print(f"  {file} entry {entry_num}: {kernel}")