import torchvision.transforms as T
from torchvision.transforms._transforms_video import ToTensorVideo

from Transforms import VideoNormalize, VideoAugment, Permute, VideoRandAugment


def generate_transforms(channels: int, video_size: tuple[int, int]):
    if channels==1:
        dataset_mean = [0.55871]
        dataset_std = [0.151]

        train_transform = T.Compose(
            [
                ToTensorVideo(),  # C, T, H, W
                VideoAugment(size=video_size),
                VideoNormalize(mean=dataset_mean, std=dataset_std)
            ]
        )

        test_transform = T.Compose(
            [
                ToTensorVideo(),
                T.Resize(size=video_size, antialias=True),
                VideoNormalize(mean=dataset_mean, std=dataset_std),
            ]
        )
    elif channels==3:
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        train_transform = T.Compose(
            [
                ToTensorVideo(),  # C, T, H, W
                Permute(dims=[1, 0, 2, 3]),  # T, C, H, W
                VideoRandAugment(magnitude=10, num_ops=2),  # C, T, H, W
                T.RandomChoice([
                    T.Resize(size=video_size, antialias=False),
                    T.Resize(size=video_size, antialias=True),
                ]),
                VideoNormalize(mean=imagenet_mean, std=imagenet_std)
            ]
        )

        test_transform = T.Compose(
            [
                ToTensorVideo(),
                T.Resize(size=video_size, antialias=True),
                VideoNormalize(mean=imagenet_mean, std=imagenet_std),
            ]
        )
    else:
        raise ValueError("Only 1 or 3 channels are supported")
    return train_transform, test_transform