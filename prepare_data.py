import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='My App.')
    parser.add_argument('--source_file', type=str, default=None)
    parser.add_argument('--target_file', type=str, default=None)
    parser.add_argument('--index_col', type=int, default=None)
    parser.add_argument('--wrapper_col', type=str, default=None)
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--end_index', type=int, default=-1)
    args = parser.parse_args()

    df = pd.read_csv(args.source_file, index_col=args.index_col)
    df = df.iloc[args.start_index:args.end_index]
    
    lines = df.to_json(orient='records', lines=True)
    
    if args.wrapper_col is not None:
        lines = "\n".join(f'{{"{args.wrapper_col}":{line}}}' for line in lines.splitlines())
    
    with open(args.target_file, 'w+') as f:
        f.write(lines)
    
if __name__ == '__main__':
    main()