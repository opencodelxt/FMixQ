from aitodpycocotools.coco import COCO
from aitodpycocotools.cocoeval import COCOeval
 
coco_true = COCO(annotation_file="/home1/wjs/datasets/AI-TOD/annotations/aitod_test_v1.json")
coco_pre = coco_true.loadRes("/home1/wjs/rtdetr-3/rtdetr_pytorch/output/rtdetr_r50vd_hgs_aitod_1_temp_1/detections.json")
cocoevaluator = COCOeval(cocoGt = coco_true, cocoDt = coco_pre, iouType = "bbox")
cocoevaluator.evaluate()
cocoevaluator.accumulate()
cocoevaluator.summarize()
