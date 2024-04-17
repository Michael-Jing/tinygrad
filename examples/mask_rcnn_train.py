# parameters to expose, use getenv
# data loader with Queues 
# A loop to loop over batches
# Models and loss functions
# use a optimizer to step back

from multiprocessing import Process, Queue
import traceback
from tinygrad.helpers import getenv
from tqdm import trange

import os 
import json 
import cv2
import numpy as np

from tinygrad.tensor import Tensor



def get_balloon_dicts(img_dir):
  json_file = os.path.join(img_dir, "via_region_data.json")
  with open(json_file) as f:
    imgs_anns = json.load(f)

  dataset_dicts = []
  for idx, v in enumerate(imgs_anns.values()):
    record = {}

    filename = os.path.join(img_dir, v["filename"])
    height, width = cv2.imread(filename).shape[:2]

    record["file_name"] = filename
    record["image_id"] = idx
    record["height"] = height
    record["width"] = width

    annos = v["regions"]
    objs = []
    for _, anno in annos.items():
      assert not anno["region_attributes"]
      anno = anno["shape_attributes"]
      px = anno["all_points_x"]
      py = anno["all_points_y"]
      poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
      poly = [p for x in poly for p in x]

      obj = {
        "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
        "bbox_mode": "XYXY_ABS",
        "segmentation": [poly],
        "category_id": 0,
      }
      objs.append(obj)
    record["annotations"] = objs
    dataset_dicts.append(record)
  return dataset_dicts

class BatchFetcher:
  def __init__(self, dataset_dicts):
    self.dataset_dicts = dataset_dicts
    self.idx = 0

  def __call__(self, bs):
    if self.idx >= len(self.dataset_dicts):
      self.idx = 0
    res = self.dataset_dicts[self.idx:self.idx+bs]
    self.idx += bs
    missing = bs - len(res)
    if missing > 0:
      res += self.dataset_dicts[:missing]
      self.idx = missing
    return res

if __name__ == "__main__":
  BS = getenv("BS", 2)
  steps = getenv("STEPS", 2)
  DATA_DIR = getenv("DATA_DIR", "balloon")
  dataset_dicts = get_balloon_dicts(DATA_DIR)
  fetcher = BatchFetcher(dataset_dicts)


  def loader(q):
    while 1:
      try:
        q.put(fetcher(BS))
      except Exception:
        traceback.print_exc()
  q = Queue(16)
  for i in range(2):
    p = Process(target=loader, args=(q,))
    p.daemon = True
    p.start()

  with Tensor.train():
    for i in (t := trange(steps)):
      l = q.get(True)
      print(l)
