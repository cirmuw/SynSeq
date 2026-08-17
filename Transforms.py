import random
import torch
from torchvision import transforms as Transforms
import torchvision.transforms.functional as F

class Permute:
    def __init__(self, dims):
        self.dims = dims

    def __call__(self, x):
        return x.permute(*self.dims)


class VideoRandAugment:
    def __init__(self, num_ops=2, magnitude=10):
        self.aug = Transforms.RandAugment(num_ops=num_ops, magnitude=magnitude)

    def __call__(self, video):
        # video: (T, C, H, W)
        frames = []
        for t in range(video.shape[0]):
            frame = video[t] * 255  # (C, H, W)
            frame = self.aug(frame.to(torch.uint8))
            frames.append((frame/255).to(torch.float32))
        return torch.stack(frames, dim=1)


class VideoNormalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(-1, 1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1, 1)

    def __call__(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

class VideoAugment:
    def __init__(self, size):
        self.size = size

    def __call__(self, video):
        # video: (C, T, H, W)
        C, T, H, W = video.shape

        # ---- sample ONCE per video ----
        angle = random.uniform(-5, 5)

        i, j, h, w = Transforms.RandomResizedCrop.get_params(
            video[:, 0], scale=(0.9, 1.0), ratio=(0.9, 1.1)
        )

        brightness = random.uniform(0.9, 1.1)
        contrast = random.uniform(0.9, 1.1)

        frames = []

        for t in range(T):
            x = video[:, t]  # (C, H, W)

            # ---- geometric (consistent) ----
            x = F.resized_crop(x, i, j, h, w, self.size, antialias=True)
            x = F.rotate(x, angle)


            # ---- intensity (consistent across video) ----
            x = F.adjust_brightness(x, brightness)
            x = F.adjust_contrast(x, contrast)

            frames.append(x)

        return torch.stack(frames, dim=1)