

class Counter:

    def __init__(self):
        self.index = 0
        self.log_list = []

    def reset(self):
        self.index = 0
        self.log_list.clear()

    def log(self, class_name, function_name, action):
        self.index += 1
        step = str(self.index).zfill(2)
        text = f"debug_{step}_{class_name}_{function_name}_{action}"
        self.log_list.append(text)
        return text
