import yaml
import torch
import argparse
import timeit
import numpy as np

from torch.utils import data
from tqdm import tqdm


from ptsemseg.models import get_model
from ptsemseg.loader import get_loader
from ptsemseg.metrics import runningScore
from ptsemseg.utils import convert_state_dict

torch.backends.cudnn.benchmark = False


def validate(cfg, args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Dataloader
    data_loader = get_loader(cfg["data"]["dataset"])
    data_path = cfg["data"]["path"]

    loader = data_loader(
        data_path,
        split=cfg["data"]["test_split"],
        is_transform=True,
        img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
    )

    n_classes = loader.n_classes

    valloader = data.DataLoader(loader, batch_size=cfg["training"]["batch_size"], num_workers=8)
    running_metrics = runningScore(n_classes)
    score, class_iou = running_metrics.get_scores()

    # Setup Model

    model = get_model(cfg["model"], n_classes).to(device)
    state = convert_state_dict(torch.load(args.model_path)["model_state"])
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    for i, (images, labels) in tqdm(enumerate(valloader)):
        start_time = timeit.default_timer()

        images = images.to(device)
        outputs = model(images)
        pred = outputs.data.max(1)[1].cpu().numpy()

        gt = labels.data.cpu().numpy()

        if args.measure_time:
            elapsed_time = timeit.default_timer() - start_time
            print(
                "Inference time \
                  (iter {0:5d}): {1:3.5f} fps".format(
                    i + 1, pred.shape[0] / elapsed_time
                )
            )
        running_metrics.update(gt, pred)
        score, class_iou = running_metrics.get_scores()
        print(score)
    

    score, class_iou = running_metrics.get_scores()

    for k, v in score.items():
        print(k, v)

    for i in range(n_classes):
        print(i, class_iou[i])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparams")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/fcn8s_pascal.yml",
        help="Config file to be used",
    )
    parser.add_argument(
        "--model_path",
        nargs="?",
        type=str,
        default="fcn8s_pascal_1_26.pkl",
        help="Path to the saved model",
    )
    parser.add_argument(
        "--measure_time",
        dest="measure_time",
        action="store_true",
        help="Enable evaluation with time (fps) measurement |\
                              True by default",
    )
    parser.add_argument(
        "--no-measure_time",
        dest="measure_time",
        action="store_false",
        help="Disable evaluation with time (fps) measurement |\
                              True by default",
    )
    parser.set_defaults(measure_time=True)

    parser.add_argument(
        "--offline_res",
        nargs="?",
        type=str,
        default="",
        help="Path to folder with offline results",
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    #need to reduce batchsize to prevent cuda errors
    cfg['training']['batch_size'] = 1

    validate(cfg, args)
