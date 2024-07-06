

class Setting:

    def __init__(self, debug_mode):
        self.debug_mode = debug_mode

        # line flow 用於校正小角度的傾斜，預設打開
        self.enable_line_flow = True

        # v2 會考慮是否有被 table 的線切到
        self.enable_orientation_detect_v2 = False
        self.orientation_detect_enable_blur = False
        self.orientation_detect_auto_canny_base = 1.0
        self.orientation_detect_auto_canny_sigma = 1.0
        self.orientation_detect_calculate_count_min_area = 225
        self.orientation_detect_calculate_count_max_area = 2500
        self.output_size = 1920

        self.rotation_lineflow_adaptive_threshold_block_size = 15
