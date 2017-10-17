#python ./tools/demo_solar.py --cfg experiments/cfgs/rfcn_end2end_SP.yml --input data/demo_solar/
#python tools/demo_solar.py --cfg experiments/cfgs/rfcn_end2end_SP.yml --input ../addressdata-cache/nearmap_images/test/
python tools/demo_solar.py --cfg experiments/cfgs/rfcn_end2end_SP.yml --input solar_test/test/
python tools/demo_chimney.py --gpu 0 --cfg experiments/cfgs/rfcn_end2end.yml --net models/chimney_streetview/ResNet-101/rfcn_end2end/test_agnostic.prototxt --model output/rfcn_end2end/chimney_streetview/resnet101_rfcn_iter_40000.caffemodel --input data/chimney_streetview/test/
