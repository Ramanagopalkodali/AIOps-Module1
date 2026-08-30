# Q3 - DVC Rollback Demonstration

## Commands used
git checkout v1
dvc checkout
wc -l Q3_DVC/data/file_list.csv
git checkout main
dvc checkout

## Output proving rollback matches v1

$ git checkout v1
Note: switching to 'v1'.
HEAD is now at 8d5fdc0 Q3: data v1 (1800 rows + header)

$ dvc checkout
M       Q3_DVC/data/

$ wc -l Q3_DVC/data/file_list.csv
1801 Q3_DVC/data/file_list.csv

The row count (1801 lines = 1800 data rows + 1 header) matches
the original v1 dataset exactly, confirming a successful rollback.

$ git checkout main
$ dvc checkout
