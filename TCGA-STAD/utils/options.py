import argparse


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description="configurations for response prediction")
    parser.add_argument("--excel_file", type=str, help="path to csv file")
    parser.add_argument("--modal", type=str, default="WSI", help="required modality")

    # Checkpoint + Misc. Pathing Parameters
    parser.add_argument("--seed", type=int, default=1, help="random seed for reproducible experiment (default: 1)")
    parser.add_argument("--log_data", action="store_true", default=True, help="log data using tensorboard")
    parser.add_argument("--evaluate", action="store_true", dest="evaluate", help="evaluate model on test set")
    parser.add_argument("--resume", type=str, default="", metavar="PATH", help="path to latest checkpoint (default: none)")

    # Model Parameters.
    parser.add_argument("--model", type=str, default="meanmil", help="type of model (default: meanmil)")
    parser.add_argument("--fusion", type=str, default=None, help="method of feature fusion")
    parser.add_argument("--layers", type=int, default=None, help="number of transformer layers")
    parser.add_argument("--n_features", type=int, default=1024, help="dimension of patche features")
    parser.add_argument("--keep_missing", action="store_true", default=False, help="keep modality-incomplete data")

    # Optimizer Parameters + Survival Loss Function
    parser.add_argument("--optimizer", type=str, choices=["Adam", "AdamW", "RAdam", "PlainRAdam", "Lookahead"], default="Adam")
    parser.add_argument("--scheduler", type=str, choices=["exp", "step", "plateau", "cosine"], default="cosine")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--num_epoch", type=int, default=20, help="maximum number of epochs to train (default: 20)")
    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate (default: 0.0001)")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--loss", type=str, default="nll_surv", help="slide-level classification loss function (default: ce)")

    # Hyperparameters
    parser.add_argument("--alpha", type=float, default=None, help="weight balance for DIV")
    parser.add_argument("--beta", type=float, default=None, help="weight balance for KD")

    args = parser.parse_args()
    return args
