./tools/train_net.py --gpu 0 --solver models/solar_panel/ResNet-101/rfcn_end2end/solver.prototxt --weights data/imagenet_models/ResNet-101-model.caffemodel --imdb solar_panel_train --iters 40000 --cfg experiments/cfgs/rfcn_end2end.yml
./tools/train_net.py --gpu 0 --solver models/chimney_streetview/ResNet-101/rfcn_end2end/solver.prototxt --weights data/imagenet_models/ResNet-101-model.caffemodel --imdb chimney_streetview_train --iters 40000 --cfg experiments/cfgs/rfcn_end2end.yml
./tools/train_net.py --gpu 0 --solver models/chimney_streetview/ResNet-101/rfcn_end2end/solver.prototxt --weights data/imagenet_models/ResNet-101-model.caffemodel --imdb chimney_streetview_train --iters 40000 --cfg experiments/cfgs/rfcn_end2end_chimney.yml 2>&1 | tee ./outputfile.log
