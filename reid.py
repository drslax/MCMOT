from torchreid.data.transforms import build_transforms
from PIL import Image
import io
import torchreid
import torch
from torchreid import metrics


class REID:
    def __init__(self):
        self.use_gpu = torch.cuda.is_available()
        self.model = torchreid.models.build_model(
            name='resnet50',
            num_classes=1,  # human
            loss='softmax',
            pretrained=True,
            use_gpu=self.use_gpu
        )
        # Load the model file into a buffer
        with open('model_data/models/model.pth', 'rb') as f:
            buffer = io.BytesIO(f.read())
        state_dict = torch.load(buffer)
        if self.use_gpu:
            self.model = self.model.cuda()
        _, self.transform_te = build_transforms(
            height=256, width=128,
            random_erase=False,
            color_jitter=False,
            color_aug=False
        )
        self.dist_metric = 'euclidean'
        self.model.eval()

    def _extract_features(self, input):
        self.model.eval()
        return self.model(input)

    def features_(self, imgs):
        f = []
        for img in imgs:
            try:
                img = Image.fromarray(img.astype('uint8')).convert('RGB')
                img = self.transform_te(img)
                img = torch.unsqueeze(img, 0)
                if self.use_gpu:
                    img = img.cuda()
                features = self._extract_features(img)
                features = features.data.cpu()  # tensor shape=1x2048
                f.append(features)
            except:
                pass
        f = torch.cat(f, 0)
        return f

    def compute_distance(self, qf, gf):
        distmat = metrics.compute_distance_matrix(qf, gf, self.dist_metric)
        # print(distmat.shape)
        return distmat.numpy()


if __name__ == '__main__':
    reid = REID()
