import cv2
from cv2 import dnn_superres
import numpy as np
import os
import re
import urllib.request


_MODEL_FACTORS = {
  'edsr': (2, 3, 4),
  'lapsrn': (8,),
  'fsrcnn': (2, 3, 4),
}
_MODEL_TO_FILE = {
  'edsr': 'https://raw.githubusercontent.com/Saafke/EDSR_Tensorflow/master/models/EDSR_x{factor}.pb',
  'lapsrn': 'https://raw.githubusercontent.com/fannymonori/TF-LapSRN/master/export/LapSRN_x{factor}.pb',
  'fsrcnn': 'https://raw.githubusercontent.com/Saafke/FSRCNN_Tensorflow/master/models/FSRCNN_x{factor}.pb',
}


class Obj(object):

  def __init__(self, *args, **kwargs):
    for d in list(args) + [kwargs]:
      for k, v in d.items():
        setattr(self, k, v)


def _select_factor(factor, factors):
  for f in factors:
    if f >= factor:
      return f

  return factors[-1]


def _get_scale_rounds(x_size, y_size, x_target, y_target, model='edsr'):
  factors = sorted(_MODEL_FACTORS[model])
  xs, ys = x_size, y_size
  while xs < x_target or ys < y_target:
    x_factor = _select_factor(x_target / xs, factors) if xs < x_target else 1
    y_factor = _select_factor(y_target / ys, factors) if ys < y_target else 1
    factor = max(x_factor, y_factor)
    if factor <= 1:
      break
    xs, ys = xs * factor, ys * factor

    yield Obj(model=model, factor=factor)


def _get_model_path(model, factor, cache_dir=None):
  if cache_dir is None:
    cache_dir = os.path.join(os.getenv('HOME'), '.cache', 'SuperRes')
  if not os.path.isdir(cache_dir):
    os.makedirs(cache_dir)

  url = _MODEL_TO_FILE[model].format(factor=factor)
  fname = re.match(r'.*/([^/]+)$', url).group(1)
  mpath = os.path.join(cache_dir, fname)
  if not os.path.isfile(mpath):
    with urllib.request.urlopen(url) as resp:
      data = resp.read()
    with open(mpath, mode='wb') as f:
      f.write(data)

  return mpath


def _split_image(img, tile_size):
  y_size, x_size, _ = img.shape
  for x in range(0, x_size, tile_size.x):
    xt = min(x_size - x, tile_size.x)
    for y in range(0, y_size, tile_size.y):
      yt = min(y_size - y, tile_size.y)

      yield Obj(x=x, y=y, img=img[y: y + yt, x: x + xt])


def _best_step(size, step, mrem=None):
  xrem = mrem or int(step // 3)
  rem = int(size % step)
  while rem != 0 and rem < xrem:
    step -= max(1, int((xrem - rem) // (size / step)))
    xrem = mrem or int(step // 3)
    rem = int(size % step)

  return step


def _get_tile_size(img, max_size):
  y_size, x_size, _ = img.shape

  return Obj(x=_best_step(x_size, max_size), y=_best_step(y_size, max_size))


def super_scale_step(img, model, factor, max_size=1000):
  model_path = _get_model_path(model, factor)

  sr = dnn_superres.DnnSuperResImpl_create()

  sr.readModel(model_path)
  sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
  sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

  sr.setModel(model, factor)

  y_size, x_size, chans = img.shape
  cimg = np.empty((y_size * factor, x_size * factor, chans), dtype=img.dtype)

  for simg in _split_image(img, _get_tile_size(img, max_size)):
    print(f'Upsampling image at ({simg.x}, {simg.y})')
    uimg = sr.upsample(simg.img)

    ys, xs, _ = uimg.shape
    xpos, ypos = simg.x * factor, simg.y * factor
    cimg[ypos: ypos + ys, xpos: xpos + xs, :] = uimg

  return cimg


def _get_scale_specs(x_size, y_size, scale_spec):
  if isinstance(scale_spec, (list, tuple)):
    xs, ys = scale_spec
    if xs is not None:
      x_target = xs
      if ys is not None:
        y_target = ys
      else:
        y_target = int(round((x_target / x_size) * y_size))
    elif ys is not None:
      y_target = ys
      x_target = int(round((y_target / y_size) * x_size))
    else:
      raise RuntimeError(f'Either the X size or the Y size must be supplied')
  else:
    x_target, y_target = int(x_size * scale_spec), int(y_size * scale_spec)

  return x_target, y_target


def super_scale(img_source, scale_spec, model='edsr', max_size=1000):
  img = cv2.imread(img_source) if isinstance(img_source, str) else img_source
  y_size, x_size, chans = img.shape
  print(f'Image: {x_size}x{y_size} ({chans} colors)')

  x_target, y_target = _get_scale_specs(x_size, y_size, scale_spec)
  for r in _get_scale_rounds(x_size, y_size, x_target, y_target, model=model):
    print(f'Scale Round: {r.model} x {r.factor}')
    img = super_scale_step(img, r.model, r.factor, max_size=max_size)

  y_size, x_size, _ = img.shape
  if x_target < x_size or y_target < y_size:
    print(f'Scaling down ({x_size}x{y_size}) -> ({x_target}x{y_target})')
    img = cv2.resize(img, (x_target, y_target), interpolation=cv2.INTER_AREA)

  return img

