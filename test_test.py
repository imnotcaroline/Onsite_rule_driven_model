from parse_and_visualize import process_one_file

road_path = "onsite/outputB/test/0_131_straight_straight_141/0_131_straight_straight_141.xodr"

road_fig ,road_ax= process_one_file(road_path)

road_fig.savefig('my_plot.png')