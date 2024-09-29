import os
import torch
from torch.utils.data import DataLoader
from torch import optim
from Src import models
from utils.fit import Fit
from utils import utils
import data_loader
from Src import config

model_use = config.config()


def run(model):
    os.makedirs(args.save_weights_path, exist_ok=True)
    if len(args.device) > 1:
        model = torch.nn.DataParallel(model.to('cuda:0'), device_ids=[0, 1], output_device=0)
    else:
        model = model.to(device=args.device[0])

    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2f" % total)
    pg = utils.get_params_groups(model, args.weight_decay)

    optimizer = optim.AdamW(pg, lr=1e-4, weight_decay=args.weight_decay)
    train_data, val_data, _ = data_loader.get_data_path()
    dataloaders = DataLoader(data_loader.Dataset(train_data, (args.image_size, args.image_size)),
                             batch_size=args.batch_size,
                             shuffle=True, num_workers=4)
    dataloaders_val = DataLoader(data_loader.Dataset(val_data, (args.image_size, args.image_size)),
                                 batch_size=args.val_batch_size,
                                 shuffle=False, num_workers=4)
    Fit(
        model,
        args,
        optimizer,
        dataloaders,
        dataloaders_val,
    ).fit()


if __name__ == "__main__":
    args = utils.get_parse()
    n = args.model_use
    # for n in [10, 11, 12, 13]:
    args.save_weights_path = './weights/' + str(n) + '/'
    config_data = model_use.model_config[n]
    model = config_data['model']
    run(model)
