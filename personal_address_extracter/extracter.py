import os
import warnings

os.environ["PADDLE_OCR_BASE_DIR"] = "D:\paddleOCR"
warnings.filterwarnings("ignore", category=UserWarning)

from paddleocr import PaddleOCR
from rich import print as rprint

ocr_worker = PaddleOCR(use_angle_cls=True, show_log=False, lang="ch")

TAG_NAME = "姓名"
TAG_SEX = "性别"
TAG_ID = "公民身份证号"
TAG_BIRTH = "出生日期"
TAG_ADDR = "户籍地址"

SAME_LINE_Y_MAX_PIXEL_OFFSET = 50
RESULT_COORD_X_OFFSET = 1
RESULT_COORD_Y_OFFSET = 0

class coord:
    def __init__(self):
        self.x = -1
        self.y = -1
        return

    def __init__(self, coord: list):
        self.x = coord[RESULT_COORD_X_OFFSET]
        self.y = coord[RESULT_COORD_Y_OFFSET]
        return

    def empty(self):
        return self.x == -1 and self.y == -1

def perform_ocr(image_path: str):
    if not os.path.isfile(image_path):
        rprint(f"[red]Error: image `{image_path}` doesn't exist.")
        return []
    
    try:
        result = ocr_worker.ocr(image_path)
    except Exception as ex:
        rprint(f"[red]Error: failed to perform OCR on image `{image_path}`.[/]\n", ex)

    return result[0]

def check_ocr_result(result) -> bool:
    if len(result) < 1:
        rprint("[red]Error: result length less than 1.")
        return False
    else:
        tag_name_found = False
        tag_sex_found = False
        tag_id_found = False
        tag_birth_found = False
        tag_addr_found = False

        for line in result:
            recognized_text = line[1][0]
            if TAG_NAME in recognized_text: tag_name_found = True
            elif TAG_SEX in recognized_text: tag_sex_found = True
            elif TAG_ID in recognized_text: tag_id_found = True
            elif TAG_BIRTH in recognized_text: tag_birth_found = True
            elif TAG_ADDR in recognized_text: tag_addr_found = True
            else: continue

        if not (tag_name_found and tag_sex_found and tag_id_found and tag_birth_found and tag_addr_found):
            rprint("[red]Error: not all necessary tag found.")
            return False
    return True

def find_data_with_tag(result, tag: str) -> tuple[str, float]:
    tag_coord_uo: coord = {}
    for line in result:
        recognized_text = line[1][0]
        if recognized_text == tag:
            tag_coord_uo = coord(line[0][0])
            # coord order: lu -> lo -> ro -> ru
            # TODO: orders of coords may be changed in different models, a reorder may be necessary
            break
    if tag_coord_uo.empty():
        rprint(f"[red]Error: tag `{tag}` not found in OCR result.")
        return ("", -1)
    
    closest_line = []
    
    for line in result:
        recognized_text = line[1][0]
        if recognized_text == tag:
            continue
        # skip tag itself
        
        this_coord_uo = coord(line[0][0])
        if abs(this_coord_uo.y - tag_coord_uo.y) < SAME_LINE_Y_MAX_PIXEL_OFFSET:
            if len(closest_line) != 0:
                if coord(closest_line[0][0]).x <= this_coord_uo.x:
                    continue
                # closest line until now has a closer position to tag, skip this line
            # closest line not empty, judge which one is closer
            closest_line = line
    if len(closest_line) == 0:
        rprint(f"[red]Error: no data assigned to tag `{tag}` was found.")
        return ("", -1)
    
    return (closest_line[1][0], closest_line[1][1])

def main():
    return

if __name__ == "__main__":
    main()
