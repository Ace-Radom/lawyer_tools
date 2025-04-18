import os
import warnings

os.environ["PADDLE_OCR_BASE_DIR"] = "D:\paddleOCR"
warnings.filterwarnings("ignore", category=UserWarning)

from paddleocr import PaddleOCR
from rich import print as rprint
from rich.panel import Panel

ocr_worker = PaddleOCR(use_angle_cls=True, show_log=False, lang="ch")

TARGET_IMAGE_DIRNAME = "images"

TAG_NAME = "姓名"
TAG_SEX = "性别"
TAG_ID = "公民身份证号"
TAG_BIRTH = "出生日期"
TAG_ADDR = "户籍地址"

SAME_LINE_Y_MAX_PIXEL_OFFSET = 50
RESULT_COORD_X_OFFSET = 1
RESULT_COORD_Y_OFFSET = 0

class coord:
    def __init__(self) -> None:
        self.x = -1
        self.y = -1
        return

    def __init__(self, coord: list) -> None:
        self.x = coord[RESULT_COORD_X_OFFSET]
        self.y = coord[RESULT_COORD_Y_OFFSET]
        return

    def empty(self) -> bool:
        return self.x == -1 and self.y == -1
    
# extracted persional data structure in one file 
class personal_data:
    def __init__(self) -> None:
        self.name = ""
        self.sex = ""
        self.id = ""
        self.birth = ""
        self.addr = ""
        return
    
    def empty(self) -> None:
        return len(self.name) == 0 and len(self.sex) == 0 and len(self.id) == 0 and len(self.birth) == 0 and len(self.addr) == 0

def perform_ocr(image_path: str) -> list:
    if not os.path.isfile(image_path):
        rprint(f"[red]Error[/]: image `{image_path}` doesn't exist.")
        return []
    elif os.path.splitext(image_path)[1] not in [".jpg", ".jpeg", ".png"]:
        rprint(f"[red]Error[/]: image `{image_path}` has a not supported extension.")
        return []
    
    try:
        result = ocr_worker.ocr(image_path)
    except Exception as ex:
        rprint(f"[red]Error[/]: failed to perform OCR on image `{image_path}`.\n", ex)

    return result[0]

def check_ocr_result(result: list) -> bool:
    if len(result) < 1:
        rprint("[red]Error[/]: result length less than 1.")
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
            rprint("[red]Error[/]: not all necessary tag found.")
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
        rprint(f"[red]Error[/]: tag `{tag}` not found in OCR result.")
        return ("", -1)
    
    closest_line = []
    
    for line in result:
        recognized_text = line[1][0]
        if recognized_text == tag:
            continue
        # skip tag itself
        
        this_coord_uo = coord(line[0][0])
        if abs(this_coord_uo.y - tag_coord_uo.y) < SAME_LINE_Y_MAX_PIXEL_OFFSET and this_coord_uo.x > tag_coord_uo.x:
            if len(closest_line) != 0:
                if coord(closest_line[0][0]).x <= this_coord_uo.x:
                    continue
                # closest line until now has a closer position to tag, skip this line
            # closest line not empty, judge which one is closer
            closest_line = line
    if len(closest_line) == 0:
        rprint(f"[red]Error[/]: no data assigned to tag `{tag}` was found.")
        return ("", -1)
    
    return (closest_line[1][0], closest_line[1][1])

def extract_picture_personal_data(image_path: str) -> personal_data:
    result = perform_ocr(image_path)
    if not check_ocr_result(result):
        return personal_data()
    # illegal ocr result

    data = personal_data()

    data.name, acc = find_data_with_tag(result, TAG_NAME)
    if acc == -1:
        return personal_data()
    elif acc < 0.95:
        rprint("[yellow]Warning[/]: low accuracy expected on `name`.")

    data.sex, acc = find_data_with_tag(result, TAG_SEX)
    if acc == -1:
        return personal_data()
    elif acc < 0.95:
        rprint("[yellow]Warning[/]: low accuracy expected on `sex`.")

    data.id, acc = find_data_with_tag(result, TAG_ID)
    if acc == -1:
        return personal_data()
    elif acc < 0.95:
        rprint("[yellow]Warning[/]: low accuracy expected on `id`.")

    data.birth, acc = find_data_with_tag(result, TAG_BIRTH)
    if acc == -1:
        return personal_data()
    elif acc < 0.95:
        rprint("[yellow]Warning[/]: low accuracy expected on `birth`.")

    data.addr, acc = find_data_with_tag(result, TAG_ADDR)
    if acc == -1:
        return personal_data()
    elif acc < 0.95:
        rprint("[yellow]Warning[/]: low accuracy expected on `addr`.")

    return data

def main():
    os.chdir(os.path.dirname(__file__))
    if not os.path.isdir(TARGET_IMAGE_DIRNAME):
        rprint(f"[red]Error[/]: target image directory `{TARGET_IMAGE_DIRNAME}` doesn't exist.")
        return 1
    for _, _, files in os.walk(TARGET_IMAGE_DIRNAME):
        rprint(f"Total [cyan]{len(files)}[/] files found.")
        for file in files:
            rprint(f"Extracting personal datas from file `[cyan]{file}[/]`...")
            data = extract_picture_personal_data(os.path.join(TARGET_IMAGE_DIRNAME, file))
            if data.empty():
                continue
            panel_text = f" [yellow]•[/] [green]{TAG_NAME}[/]: {data.name}\n"
            panel_text += f" [yellow]•[/] [green]{TAG_SEX}[/]: {data.sex}\n"
            panel_text += f" [yellow]•[/] [green]{TAG_ID}[/]: {data.id}\n"
            panel_text += f" [yellow]•[/] [green]{TAG_BIRTH}[/]: {data.birth}\n"
            panel_text += f" [yellow]•[/] [green]{TAG_ADDR}[/]: {data.addr}"
            panel = Panel(
                panel_text,
                title=f"Datas from `[cyan]{file}[/]`"
            )
            rprint(panel)
            

if __name__ == "__main__":
    main()
