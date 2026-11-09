import json
from pathlib import Path
import cv2
import numpy as np

def _fit(image, width, height):
    return cv2.resize(image, (width,height), interpolation=cv2.INTER_AREA)

def render_dashboard(result):
    panel_w, panel_h = 900, 560
    top = _fit(result.top_view, panel_w, panel_h)
    side = _fit(result.side_view, panel_w, panel_h)
    sx, sy = panel_w/result.top_view.shape[1], panel_h/result.top_view.shape[0]
    for i,b in enumerate(result.bricks, start=1):
        p1=(int(b.x1*sx),int(b.y1*sy)); p2=(int(b.x2*sx),int(b.y2*sy))
        cv2.rectangle(top,p1,p2,(0,255,0),2)
        cv2.putText(top,str(i),(p1[0],max(18,p1[1]-4)),
                    cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
    sx2, sy2 = panel_w/result.side_view.shape[1], panel_h/result.side_view.shape[0]
    for idx,y in enumerate(result.seams, start=1):
        yy=int(y*sy2)
        cv2.line(side,(0,yy),(panel_w-1,yy),(0,165,255),2)
        cv2.putText(side,f"seam {idx}",(12,max(18,yy-4)),
                    cv2.FONT_HERSHEY_SIMPLEX,.45,(0,165,255),1,cv2.LINE_AA)
    header_h, footer_h, gap = 54, 100, 12
    canvas=np.full((header_h+panel_h+footer_h,panel_w*2+gap,3),24,np.uint8)
    cv2.putText(canvas,"TOP VIEW: brick candidates",(18,36),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
    cv2.putText(canvas,"SIDE VIEW: layer seams",(panel_w+gap+18,36),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
    canvas[header_h:header_h+panel_h,0:panel_w]=top
    canvas[header_h:header_h+panel_h,panel_w+gap:]=side
    y=header_h+panel_h
    cv2.putText(canvas,f"{result.bricks_per_layer} bricks / layer",(18,y+38),
                cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0),2)
    cv2.putText(canvas,f"x {result.layers} layers",(420,y+38),
                cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,200,255),2)
    cv2.putText(canvas,f"TOTAL: {result.total}",(panel_w+gap+18,y+38),
                cv2.FONT_HERSHEY_SIMPLEX,1.1,(0,255,255),3)
    cv2.putText(canvas,f"Sources: {result.top_count_source}; {result.layer_count_source}",
                (18,y+76),cv2.FONT_HERSHEY_SIMPLEX,.48,(210,210,210),1)
    return canvas

def save_summary(output_dir, result):
    out=Path(output_dir)
    data={
        "bricks_per_layer":result.bricks_per_layer,
        "layers":result.layers,
        "total_bricks":result.total,
        "top_count_source":result.top_count_source,
        "layer_count_source":result.layer_count_source,
        "detected_brick_boxes":[[b.x1,b.y1,b.x2,b.y2] for b in result.bricks],
        "side_seam_rows":result.seams
    }
    (out/"summary.json").write_text(json.dumps(data,indent=2),encoding="utf-8")
