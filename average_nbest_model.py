"""
Adapted from espnet2/main_funcs/average_nbest_models.py
"""
import os
import torch


def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--model-dir', type=str, default='exp/train_asr/ckpts')
    parser.add_argument('--output', type=str, default='exp/train_asr/ckpts/val.wer.ave_5best.pth')
    parser.add_argument('-n', type=int, default=5)
    return parser.parse_args()


def main():
    args = get_args()

    q = []
    for file in os.listdir(args.model_dir):
        path = os.path.join(args.model_dir, file)
        if os.path.isfile(path) and path.endswith('.pth') and file.startswith('epoch'):
            checkpoint = torch.load(path, map_location='cpu')
            model_dict = checkpoint["model_state_dict"]
            wer = checkpoint["wer"]
            q.append((wer, model_dict, file))

    q = sorted(q)[:args.n]
    print('Checkpoint WER:', {e[2]: e[0] for e in q})

    avg = None
    for wer, states, _ in q:
        if avg is None:
            avg = states
        else:
            for k in avg:
                avg[k] = avg[k] + states[k]
    for k in avg:
        if str(avg[k].dtype).startswith("torch.int"):
            # for int type, not averaged, but only accumulated.
            pass
        else:
            avg[k] = avg[k] / args.n

    torch.save({'model_state_dict': avg}, args.output)


if __name__ == '__main__':
    main()
