import argparse
import numpy as np
import importlib
import torch
from torch.utils.data import DataLoader
import wandb


from data.utils import get_dataset
from AImodels import get_model



from llf.learner import learner
from andmask import train_andmask


parser = argparse.ArgumentParser(description="Generate bias detection/ mitigation results.")


### Data
parser.add_argument("--dataset", type=str, default='cmnist',
                    help="Dataset name.")
parser.add_argument("--data_dir", type=str, default='data',
                    help="Root folder for all datasets. Complete used path is `dataset_root/dataset_name`.")
parser.add_argument("--percent", type=str, default="1pct",
                    help="Select percent of align/conflict examples in CMNIST dataset in {0.5pct, 1pct, 2pct, 5pct}")

### Model
parser.add_argument("--model", type=str, default='resnet50',
                    help="Model architecture. choose between 'MLP', 'ResNet18, ...'.")
parser.add_argument("--batch_size", type=int, default=16,
                    help="Model batch_size.")
parser.add_argument("--lr", type=float, default=1e-3,
                    help="Model learning rate.")
parser.add_argument("--weight_decay",help='weight_decay',default=0.0, type=float)
parser.add_argument("--seed", type=int, default=14,
                    help="Random seed.")

### Other params
parser.add_argument("--device", help="cuda or cpu", default='cuda', type=str)
parser.add_argument("--num_workers", type=int, default=8, help="Number of workers used in PyTorch DataLoader.")
parser.add_argument("--mitigation", type=str, default=None, help= "Mitigation methods name (lff, ldbb, debian), or train a 'vanilla' model.")
parser.add_argument("--logs", help='log directory for saving models/experiments.', default='logs')
parser.add_argument('--epochs', default=100, type=int)

### Method dependent...
parser.add_argument("--target_attr_idx", help="target_attr_idx", default= 0, type=int)
parser.add_argument("--bias_attr_idx", help="bias_attr_idx", default= 1, type=int)
#parser.add_argument("--num_steps", help="# of iterations", default= 500 * 100, type=int)
parser.add_argument("--valid_freq", help='frequency to evaluate on valid/test set', default=500, type=int)

def main():
    # Get arguments
    # global args
    args = parser.parse_args()
    print(args)

    # Seed everything
    # random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Get dataset
    # dataset, n_output = get_dataset(args.dataset_name, args.data_dir, args.dataset_percent)
    # train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    # train_features, train_labels, train_img_names = next(iter(train_dataloader))
    # print(f"Feature batch shape: {train_features.size()}")
    # print(f"Labels batch shape: {train_labels.size()}")
    # print(train_img_names)
    # label = train_labels[0]
    # print(f"Label: {label}")

    # Get model
    # model = get_model(args.model, n_output=n_output, dataset=args.dataset_name)
    # model = get_model(args.model, n_output, dataset=args.dataset_name)
    # model = model.eval()

    #Getting the config for the run 
    configs = vars(args)

    wandb.init(
    # set the wandb project where this run will be logged
    project = "bias_unlabeled",
    
    # track hyperparameters and run metadata
    config = configs

)
    # Learning From Failure
    if args.mitigation=="lff":
        learner.train(main_tag = "lff",
        dataset_tag = args.dataset,
        model_tag = args.model,
        data_dir = args.data_dir,
        log_dir = f"{args.logs}/{args.mitigation}_{args.dataset}",
        device = args.device,
        target_attr_idx = args.target_attr_idx,
        bias_attr_idx = args.bias_attr_idx,
        main_num_epochs = args.epochs,
        main_valid_freq = args.valid_freq,
        main_batch_size = args.batch_size,
        main_learning_rate = args.lr,
        main_weight_decay = args.weight_decay,
        percent = args.percent,
        num_workers=args.num_workers,
        wandb_logger = wandb
        )

    elif args.mitigation=="ldd":
        ldd = importlib.import_module("Learning-Debiased-Disentangled-master.learner")
        learnerBase = ldd.Learner(args)
        learnerBase.train_ours(args)

    elif args.mitigation=="vanilla":
        ldd = importlib.import_module("Learning-Debiased-Disentangled-master.learner")
        learnerVan = ldd.Learner(args)
        learnerVan.train_vanilla(args)

    elif args.mitigation == "andmask":
        train_andmask( main_tag = "andmask",
                    dataset_tag = args.dataset,
                    model_tag = args.model,
                    data_dir = args.data_dir,
                    # log_dir = args.logs + args.mitigation,
                    # target_attr_idx = args.target_attr_idx,
                    # bias_attr_idx = args.bias_attr_idx,
                    # main_valid_freq,
                    main_batch_size = args.batch_size,
                    # main_learning_rate,
                    # main_weight_decay,
                    percent = args.percent,
                    num_workers = args.num_workers,
                    output_dir = args.logs,
                    agreement_threshold = args.lambda_penalty,
                    weight_decay = 1e-6,
                    method = "and_mask",
                    scale_grad_inverse_sparsity = 1,
                    init_lr = args.lr,
                    random_labels_fraction = 1,
                    weight_decay_order = "before",
                    seed = 0,
                    batch_size = args.batch_size,
                    epoch = args.epoch,
                    get_dataset = get_dataset,
                    get_model = get_model
                )
    else:
        print(f"Mitigation method name provided:{args.mitigation} is not: lff, ldd, debian or vanilla (normal training).")




if __name__ == "__main__":
    main()
