import requests
import pickle
import zlib
from multiprocessing.dummy import Pool as ThreadPool
from lunar_seg import *
import cv2


device = torch.device('cuda')
segmenter = Segmenter.load_from_checkpoint(model_checkpoint.best_model_path).to(device)  # wczytanie najlepszych wag z treningu
segmenter = segmenter.eval()


input_transforms = val_dataset.transforms
output_transforms = A.Compose([
    A.CenterCrop(*val_dataset.image_size),
    A.Resize(720, 1280, interpolation=cv2.INTER_NEAREST)
])

test_base_path = Path('LunarSeg/test')
predictions_path = Path('LunarSeg/test/predictions')
predictions_path.mkdir(exist_ok=True, parents=True)

for test_image_path in (test_base_path / 'images').iterdir():
    image = cv2.imread(str(test_image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = input_transforms(image=image)['image'][None, ...]

    with torch.no_grad():
        prediction = segmenter(image.to(device)).cpu().squeeze().argmax(dim=0).numpy()

    prediction = convert_ids_to_rgb(prediction)
    prediction = cv2.cvtColor(prediction, cv2.COLOR_RGB2BGR)
    prediction = output_transforms(image=prediction)['image']

    cv2.imwrite(str(predictions_path / f'{test_image_path.stem}.png'), prediction)


suma = 0


def calculate_score(prediction_path: Path):
    prediction = cv2.imread(str(prediction_path))
    prediction = cv2.cvtColor(prediction, cv2.COLOR_BGR2RGB)
    global suma
    response = requests.post(f'http://zpo.dpieczynski.pl/{prediction_path.stem}',
                             data=zlib.compress(pickle.dumps(prediction)))
    if response.status_code == 200:
        result = response.json()
        res = sum(result.values())
        suma = suma + res
        return f'{prediction_path.name} {result}'
    else:
        return f'Error processing prediction {prediction_path.name}: {response.text}'

    return None


with ThreadPool(processes=16) as pool:
    cnt = 0
    for result in pool.imap_unordered(calculate_score, predictions_path.iterdir()):
        cnt = cnt + 1
        print(result)

suma = suma / cnt
print('SUMA: ', suma)