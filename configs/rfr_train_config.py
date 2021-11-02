import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', type=str, default='AnimeInterp_no_cupy')
    parser.add_argument('--pwc_path', type=str, default=None)#checkpoint path
    parser.add_argument('--saved_model', type=str, default=None) 
    