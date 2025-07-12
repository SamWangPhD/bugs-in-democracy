import os
import shutil

# ─── CONFIGURE ────────────────────────────────────────────────────────────────
folder_a = r"C:\Users\mahsh\Documents\GitHub\bugs-in-democracy\cleaned files\dataverse_files"
folder_b = r"C:\Users\mahsh\Documents\GitHub\bugs-in-democracy\cleaned files\dataverse_files_2025"
dest    = r"C:\Users\mahsh\Documents\GitHub\bugs-in-democracy\cleaned files\dataverse_files_only_after_2023"
# ────────────────────────────────────────────────────────────────────────────────

# Make sure the destination exists
os.makedirs(dest, exist_ok=True)

# List just the .csv files in each folder
files_a = {f for f in os.listdir(folder_a) if f.lower().endswith(".csv")}
files_b = {f for f in os.listdir(folder_b) if f.lower().endswith(".csv")}

# Find names only in one folder
unique = files_a.symmetric_difference(files_b)

print(f"Found {len(unique)} unique file(s):")
for fname in sorted(unique):
    print(" ", fname)

# Copy each unique file from whichever folder it lives in
for fname in unique:
    if fname in files_a:
        src = os.path.join(folder_a, fname)
    else:
        src = os.path.join(folder_b, fname)

    dst = os.path.join(dest, fname)
    shutil.copy2(src, dst)

print(f"\nCopied all unique files into: {dest}")
