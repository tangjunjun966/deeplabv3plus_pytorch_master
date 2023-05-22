
from tqdm import tqdm
import torch


def validate(model, loader, device, metrics):
    """Do validation and return specified samples"""
    model.eval()
    metrics.reset()

    with torch.no_grad():
        with tqdm(total=len(loader)) as pbar:
            for i, (images, labels) in enumerate(loader):
                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)
                outputs = model(images)
                preds = outputs.detach().max(dim=1)[1].cpu().numpy()
                targets = labels.cpu().numpy()
                metrics.update(targets, preds)
                pbar.set_description("val {}|{}\t".format(len(loader),i+1))
                pbar.update(1)
        score = metrics.get_results()
    return score










