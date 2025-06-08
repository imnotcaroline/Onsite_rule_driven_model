import colorsys
import os
import pickle
import xml.dom.minidom
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import transforms
from matplotlib.patches import Rectangle
from parse_and_visualize import process_one_file
# 设置全局字体 (确保只设置一次)
plt.rcParams.update({
     'font.family': 'Times New Roman',
     'font.size': 10,
     'axes.titlesize': 11,
     'axes.labelsize': 10,
     'axes.labelweight': 'bold',
     'xtick.labelsize': 9,
     'ytick.labelsize': 9,
     'legend.fontsize': 8,
     'figure.figsize': (7, 12),  #保持你原有的大小比例
     'figure.dpi': 300,
     'savefig.dpi': 600, # 提高为600，确保高质量打印
     'savefig.format': 'tiff', # 科研期刊常用格式
     'savefig.bbox': 'tight',
     'savefig.pad_inches': 0.05,
     'axes.grid': True,
     'grid.alpha': 0.2,
     'grid.linestyle': ':',
     'axes.linewidth': 0.8,
     'axes.edgecolor': '#333333',
     'xtick.direction': 'out',
     'ytick.direction': 'out',
     'xtick.major.width': 0.8,
     'ytick.major.width': 0.8,
     'xtick.minor.visible': False,
     'ytick.minor.visible': False,
     'xtick.minor.width': 0.6,
     'ytick.minor.width': 0.6,
     'mathtext.fontset': 'stix', # 使用stix渲染数学公式
     'axes.unicode_minus': True, # 确保负号显示正确
     'mathtext.rm': 'Times New Roman',
     'mathtext.it': 'Times New Roman:italic',
     'mathtext.bf': 'Times New Roman:bold',
})




class OutputVisualization:
    def __init__(self, pkl_path,road_path):
        self.pkl_path = pkl_path
        self.road_path=road_path


    def get_n_hls_colors(self, num):
        hls_colors = [(i / num, 0.7, 0.9) for i in range(num)]  # 均匀色相，固定亮度和饱和度
        rgb_colors = []
        for h, l, s in hls_colors:
            r, g, b = colorsys.hls_to_rgb(h, l, s)
            rgb_colors.append((int(r * 255), int(g * 255), int(b * 255)))
        return rgb_colors

    def get_n_hls_colors_preset(self, num):
        """
        生成 N 种预设的、视觉上更容易区分的 HLS 颜色。
        适用于小数量的颜色，提供更好的区分度。
        """
        if num <= 0:
            return []

        # 预设的、视觉上差异较大的色相值 (0.0 到 1.0 范围)
        # 例如：红、绿、蓝、黄、品红、青、橙、紫、棕、灰等
        # 这些值是经过经验验证，能提供良好视觉区分度的
        distinct_hues = [
            0.0,  # 红色
            0.333,  # 绿色
            0.666,  # 蓝色
            0.166,  # 黄色
            0.833,  # 品红色
            0.5,  # 青色
            0.083,  # 橙色
            0.75,  # 紫色
            0.125,  # 金色/棕黄色
            0.916  # 玫瑰红/粉红色
        ]

        hls_colors = []
        # 使用这些预设的色相，并固定亮度和饱和度为较好的值
        l_val = 0.5
        s_val = 0.9

        # 如果所需颜色数量小于预设数量，则只取预设中的部分
        for i in range(min(num, len(distinct_hues))):
            h = distinct_hues[i]
            hls_colors.append((h, l_val, s_val))

        # 如果 num 大于预设数量，为了满足数量要求，
        # 可以考虑回到策略1的方法，或循环使用这些预设色相并改变亮/饱和度
        # 这里我们简单地继续均匀分布并略微调整亮度和饱和度
        for i in range(len(distinct_hues), num):
            h = i / num  # 继续均匀分布色相
            l = 0.5 + 0.1 * (i % 2)  # 亮度的简单交错
            s = 0.8 + 0.1 * (i % 3)  # 饱和度的简单交错
            hls_colors.append((h, l, s))

        rgb_colors = []
        for h, l, s in hls_colors:
            r, g, b = colorsys.hls_to_rgb(h, l, s)
            rgb_colors.append((int(r * 255), int(g * 255), int(b * 255)))
        return rgb_colors

    def filter_trailing_zeros_1d(self, array):
        """过滤掉一维数组末尾连续的零值"""
        if len(array) == 0:
            return np.array([])  # 处理空数组

        # # 从后向前查找第一个非零元素的索引
        # for i in range(len(array) - 1, -1, -1):
        #     if array[i] != 0:
        #         return array[:i + 1]  # 返回从开头到该索引的所有元素
        #
        # # 如果全是零，返回空数组
        # return np.array([])
        # 找到第一个非零元素的索引
        first_non_zero_idx = -1
        for i in range(len(array)):
            if array[i] != 0:
                first_non_zero_idx = i
                break

        # 如果整个数组都是零，或者没有非零元素
        if first_non_zero_idx == -1:
            return np.array([])

        # 找到最后一个非零元素的索引
        last_non_zero_idx = -1
        for i in range(len(array) - 1, -1, -1):
            if array[i] != 0:
                last_non_zero_idx = i
                break

        # 返回从第一个非零元素到最后一个非零元素的所有数据
        return array[first_non_zero_idx: last_non_zero_idx + 1]

    def filter_trailing_zeros(self, position_array):
        """过滤掉数组末尾连续的[0, 0]行"""
        # 找到所有非零行的索引
        non_zero_indices = np.where(~np.all(position_array == 0, axis=1))[0]

        if len(non_zero_indices) == 0:
            return np.array([])  # 如果全是零，返回空数组

        # 获取第一个非零行的索引
        first_non_zero = non_zero_indices[0]
        # 获取最后一个非零行的索引
        last_non_zero = non_zero_indices[-1]

        # # 返回从开头到最后一个非零行的所有数据
        # return position_array[:last_non_zero + 1]
        # 返回从第一个非零行到最后一个非零行的所有数据
        return position_array[first_non_zero: last_non_zero + 1]

    def run(self):

        # 读取地图
        # road_path = "onsite/outputB/test/0_131_straight_straight_141/0_131_straight_straight_141.xodr"
        road_fig ,road_ax= process_one_file(self.road_path)
        # 读取数据文件
        with open(self.pkl_path, 'rb') as f:
            data = pickle.load(f)
            # 读取pkl文件中的各个agent的位置/state

        # 获取所有数据
        title = data['scene_name']  # pkl轨迹可视化图像的标题
        print("pkl图像title:", title)
        ids = data['ids']
        predict_mask = data['predict_mask']
        agent_types = data['types']
        positions = data['positions']
        shape = data['shapes']
        headings = data['headings']

        # 生成颜色列表
        # colors_generate = self.get_n_hls_colors(20)
        colors_generate = self.get_n_hls_colors_preset(20)
        colors_list = ["#{:02x}{:02x}{:02x}".format(r, g, b) for r, g, b in colors_generate]

        # 创建保存目录
        project_path=os.getcwd()
        print("PROJECT_PATH:", project_path)
        output_dir = os.path.join("onsite", "outputB", "image")
        d, ne = os.path.split(self.pkl_path)
        n, e = os.path.splitext(ne)
        desired_string = n
        filename=f"{desired_string}.pdf"
        save_path = os.path.join(output_dir, filename)
        print("save path:", save_path)
        ax = road_fig.axes[0]

        # 设置标题与坐标信息（可根据已有图灵活调整）
        ax.set_title("Vehicle Trajectory Visualization_" + title, fontsize=14)
        ax.set_xlabel("X Position (m)", fontsize=12)
        ax.set_ylabel("Y Position (m)", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_aspect('equal', adjustable='box')


        # ========== 绘制轨迹与车辆 ==========
        for index, positions_n in enumerate(positions):
            if predict_mask[index] and agent_types[index] == 0:
                filtered_positions = self.filter_trailing_zeros(positions_n)
                if len(filtered_positions) == 0:
                    continue

                agent_name = ids[index]
                agent_length = shape[index][0]
                agent_width = shape[index][1]
                color_n = colors_list[index]
                x_list = filtered_positions[:, 0].tolist()
                y_list = filtered_positions[:, 1].tolist()
                headings_n = self.filter_trailing_zeros_1d(headings[index])

                # 轨迹线
                ax.plot(x_list, y_list, linestyle='-', linewidth=15, color=color_n, alpha=0.7,
                        label=f"{agent_name}_Trajectory")

                # 最终位置车辆框
                last_x, last_y = x_list[-1], y_list[-1]
                rect_x = last_x - agent_width / 2
                rect_y = last_y - agent_length / 2
                last_heading = np.rad2deg(headings_n[-1]) + 90
                t = transforms.Affine2D().rotate_deg_around(last_x, last_y, last_heading) + ax.transData
                vehicle_rect = Rectangle(
                    (rect_x, rect_y), agent_width, agent_length,
                    facecolor=color_n, edgecolor='black', alpha=0.9,
                    label=f"{agent_name}_Size({agent_length:.1f}x{agent_width:.1f})",
                    transform=t
                )
                ax.add_patch(vehicle_rect)

                # 起点终点标记
                ax.scatter(x_list[0], y_list[0], color='green', s=300, marker='o', label=f"{agent_name}_Start")
                ax.scatter(x_list[-1], y_list[-1], color='red', s=300, marker='o', label=f"{agent_name}_End")

        # ========== 右侧添加图例 ==========

        # 可以使用 inset_axes 插入子图，也可以使用 gridspec 手动创建右侧图例位置
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        legend_ax = inset_axes(ax, width="50%", height="100%", loc='center left',
                               bbox_to_anchor=(1.02, 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0)

        legend_ax.axis('off')
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = {}
        for h, l in zip(handles, labels):
            if l not in unique_labels:
                unique_labels[l] = h
        legend_ax.legend(
            unique_labels.values(), unique_labels.keys(),
            loc='center left',  fontsize=125,frameon=False
        )

        # ========== 保存图像 ==========
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存至: {save_path}")

        # 创建画布和GridSpec布局
        # fig = plt.figure(figsize=(15, 8))  # 增加宽度以容纳图例
        # fig = road_fig  # 增加宽度以容纳图例
        # gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1])  # 左侧4份宽度用于绘图，右侧1份用于图例
        #
        # # 在左侧子图上绘制轨迹
        # ax = fig.add_subplot(gs[0, 0])
        # ax.set_title("Vehicle Trajectory Visualization_" + title, fontsize=14)
        # ax.set_xlabel("X Position (m)", fontsize=12)
        # ax.set_ylabel("Y Position (m)", fontsize=12)
        # ax.grid(True, linestyle='--', alpha=0.7)
        # ax.set_aspect('equal', adjustable='box')
        #
        # # 绘制轨迹代码保持不变...
        # for index, positions_n in enumerate(positions):
        #     if predict_mask[index] and agent_types[index] == 0:
        #         filtered_positions = self.filter_trailing_zeros(positions_n)
        #         if len(filtered_positions) == 0:
        #             continue
        #         agent_name = ids[index]  # 获取agent名称
        #         agent_length = shape[index][0]  # 获取agent的长
        #         agent_width = shape[index][1]  # 获取agent的宽
        #         color_n = colors_list[index]  # 获取agent的绘制颜色
        #         x_list = filtered_positions[:, 0].tolist()  # 获取x列表
        #         y_list = filtered_positions[:, 1].tolist()  # 获取y列表
        #         headings_n = self.filter_trailing_zeros_1d(headings[index])  # 获取headings列表
        #
        #         # 绘制轨迹线
        #         ax.plot(x_list, y_list, linestyle='-', linewidth=1.5, color=color_n, alpha=0.7,
        #                 label=f"{agent_name}_Trajectory")
        #
        #         # 绘制最后位置的车辆矩形
        #         last_x, last_y = x_list[-1], y_list[-1]
        #         # 计算矩形左下角坐标，使其中心在 (last_x, last_y)
        #         rect_x = last_x - agent_width / 2
        #         rect_y = last_y - agent_length / 2
        #         # 获取最后的 heading（角度制）
        #         last_heading = np.rad2deg(headings_n[-1])+90
        #         # 创建旋转变换（绕中心 last_x, last_y 旋转）
        #         t = transforms.Affine2D().rotate_deg_around(last_x, last_y, last_heading) + ax.transData
        #         # 创建旋转后的矩形
        #         vehicle_rect = Rectangle(
        #             (rect_x, rect_y), agent_width, agent_length,
        #             facecolor=color_n, edgecolor='black', alpha=0.9,
        #             label=f"{agent_name}_Size({agent_length:.1f}x{agent_width:.1f})",
        #             transform=t  # 应用旋转变换
        #         )
        #
        #         # 添加到图上
        #         ax.add_patch(vehicle_rect)
        #
        #         # 标注起点和终点
        #         ax.scatter(x_list[0], y_list[0], color='green', s=50, marker='o', label=f"{agent_name}_Start")
        #         ax.scatter(x_list[-1], y_list[-1], color='red', s=50, marker='o', label=f"{agent_name}_End")
        #
        # # 不直接在图形上显示图例
        # handles, labels = ax.get_legend_handles_labels()
        # unique_labels = {}
        # for h, l in zip(handles, labels):
        #     if l not in unique_labels:
        #         unique_labels[l] = h
        #
        # # 在右侧子图上添加图例
        # legend_ax = fig.add_subplot(gs[0, 1])
        # legend_ax.axis('off')  # 隐藏坐标轴
        # legend_ax.legend(
        #     unique_labels.values(),
        #     unique_labels.keys(),
        #     loc='center left',
        #     fontsize=9,
        #     frameon=False  # 无边框
        # )
        #
        # plt.tight_layout()
        # plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # print(f"图像已保存至: {save_path}")


if __name__ == '__main__':
    # pkl_path= r"onsite/outputB/results/good_result/0_131_straight_straight_141_output.pkl"
    pkl_path=r"output_800/0_36_straight_straight_39_output.pkl"
    road_path=r"onsite/outputB/test/0_36_straight_straight_39/0_36_straight_straight_39.xodr"
    output_visual = OutputVisualization(pkl_path,road_path)
    output_visual.run()
