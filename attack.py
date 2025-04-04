import os
import torch
import torchvision.utils as vutils
from tqdm import tqdm
from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import check_img_size
from utils.loss import ComputeLoss
from utils.torch_utils import select_device


def i_fgsm_attack(model, img, target, loss_fn, epsilon=8/255, alpha=2/255, iters=10):
    model.eval()
    ori_img = img.clone().detach()
    perturbed = ori_img.clone().detach().requires_grad_(True)

    for _ in range(iters):
        pred = model(perturbed)[0]
        loss, _ = loss_fn(pred, target)
        model.zero_grad()
        loss.backward()
        grad = perturbed.grad.data

        perturbed = perturbed + alpha * grad.sign()
        perturb = torch.clamp(perturbed - ori_img, min=-epsilon, max=epsilon)
        perturbed = torch.clamp(ori_img + perturb, min=0, max=1).detach().requires_grad_(True)

    return perturbed.detach()

def main():
    weights = 'best.pt'  # 权重路径
    data_yaml = 'data/voc2007.yaml'  # VOC 数据配置
    save_dir = 'runs/adv_results'  # 保存路径
    batch_size = 32
    img_size = 640
    device = select_device('0')  # 改成 'cpu' 如果没有GPU

    os.makedirs(save_dir, exist_ok=True)

    model = attempt_load(weights, map_location=device)
    model.eval()

    img_size = check_img_size(img_size, s=model.stride.max())
    dataloader = create_dataloader(data_yaml, img_size, batch_size, 32, pad=0.5, rect=True)[0]

    compute_loss_fn = lambda pred, target: ComputeLoss(pred, target, model)

    for idx, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc="Generating adversarial images")):
        imgs = imgs.to(device).float() / 255.0
        targets = targets.to(device)

        adv_imgs = []
        for i in range(imgs.size(0)):
            img = imgs[i].unsqueeze(0)
            tgt = targets[targets[:, 0] == i]
            adv_img = i_fgsm_attack(model, img, tgt, compute_loss_fn)
            adv_imgs.append(adv_img)

            filename = os.path.basename(paths[i])
            save_path = os.path.join(save_dir, f"adv_{filename}")
            vutils.save_image(adv_img, save_path)

    print(f"\n✅ 所有攻击图像已保存到: {save_dir}")

if __name__ == "__main__":
    main()
