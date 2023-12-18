import math
import numpy as np
import dataclasses
import matplotlib.pyplot as plt
from typing import Tuple, Iterable
from pycpd import RigidRegistration

MAP_SIZE = 15 # Размер комнаты
SCAN_STEP = 5 # Шаг сканера
SCAN_ANGLE_RANGE = 360 # Область сканирования в градусах
MAX_SCAN_RANGE = 999 # Максимальная дистанция сканера

@dataclasses.dataclass
class RobotState:
    x: float
    y: float
    theta: float

@dataclasses.dataclass
class Room:
    map_as_2d_array: np.ndarray
    map_size: int
    obstacle_positions: Iterable[Tuple[float, float]]

@dataclasses.dataclass
class Transformation:
    s: float
    R: np.array
    t: Tuple[float, float]

@dataclasses.dataclass
class ScenarioSetup:
    obstacle_coords: Iterable[Tuple[float, float]]
    robot_states: Iterable[Tuple[RobotState, RobotState]]

# Комната 1
obstacle_coords1 = [(9, 1), (10, 1), (10, 2), (11, 2), (12, 2), (12, 3), (12, 4), (13, 4), (13, 5)]

robot_state1_i1_start = RobotState(x=4, y=6, theta=0)
robot_state1_i1_end = RobotState(x=8, y=10, theta=0)

robot_state1_i2_start = RobotState(x=2, y=12, theta=0)
robot_state1_i2_end = RobotState(x=6, y=8, theta=0)

robot_state1_i3_start = RobotState(x=7, y=7, theta=0)
robot_state1_i3_end = RobotState(x=6, y=8, theta=math.radians(30))

robot_state1_i4_start = RobotState(x=7, y=7, theta=0)
robot_state1_i4_end = RobotState(x=8, y=8, theta=math.radians(30))

robot_state1_i5_start = RobotState(x=7, y=7, theta=0)
robot_state1_i5_end = RobotState(x=7, y=7, theta=math.radians(180))

scenario1 = ScenarioSetup(
    obstacle_coords=obstacle_coords1,
    robot_states=[
        (robot_state1_i1_start, robot_state1_i1_end),
        (robot_state1_i2_start, robot_state1_i2_end),
        (robot_state1_i3_start, robot_state1_i3_end),
        (robot_state1_i4_start, robot_state1_i4_end),
        (robot_state1_i5_start, robot_state1_i5_end),
    ]
)

# Комната 2
obstacle_coords2 = [(5, 7), (9, 7)]

robot_state2_i1_start = RobotState(x=7, y=3, theta=0)
robot_state2_i1_end = RobotState(x=7, y=1, theta=0)

robot_state2_i2_start = RobotState(x=2, y=2, theta=0)
robot_state2_i2_end = RobotState(x=12, y=2, theta=0)

robot_state2_i3_start = RobotState(x=7, y=3, theta=math.radians(45))
robot_state2_i3_end = RobotState(x=7, y=3, theta=math.radians(135))

robot_state2_i4_start = RobotState(x=4, y=2, theta=0)
robot_state2_i4_end = RobotState(x=6, y=2, theta=math.radians(30))

robot_state2_i5_start = RobotState(x=10, y=12, theta=0)
robot_state2_i5_end = RobotState(x=6, y=12, theta=0)

scenario2 = ScenarioSetup(
    obstacle_coords=obstacle_coords2,
    robot_states=[
        (robot_state2_i1_start, robot_state2_i1_end),
        (robot_state2_i2_start, robot_state2_i2_end),
        (robot_state2_i3_start, robot_state2_i3_end),
        (robot_state2_i4_start, robot_state2_i4_end),
        (robot_state2_i5_start, robot_state2_i5_end),
    ]
)
# Комната 3
obstacle_coords3 = [
    (1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (5, 4), (5, 3), (5, 2), (5, 1),
    (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (10, 5), (11, 5), (12, 5), (13, 5),
    (1, 10), (2, 10), (3, 10), (4, 10), (5, 10), (5, 11), (5, 12), (5, 13),
    (9, 13), (9, 12), (9, 11), (9, 10), (10, 10), (11, 10), (12, 10), (13, 10)
]

robot_state3_i1_start = RobotState(x=3, y=7, theta=0)
robot_state3_i1_end = RobotState(x=11, y=7, theta=0)

robot_state3_i2_start = RobotState(x=5, y=7, theta=0)
robot_state3_i2_end = RobotState(x=7, y=7, theta=0)


robot_state3_i3_start = RobotState(x=7, y=7, theta=math.radians(0))
robot_state3_i3_end = RobotState(x=7, y=7, theta=math.radians(60))

robot_state3_i4_start = RobotState(x=7, y=7, theta=math.radians(0))
robot_state3_i4_end = RobotState(x=7, y=9, theta=math.radians(60))

scenario3 = ScenarioSetup(
    obstacle_coords=obstacle_coords3,
    robot_states=[
        (robot_state3_i1_start, robot_state3_i1_end),
        (robot_state3_i2_start, robot_state3_i2_end),
        (robot_state3_i3_start, robot_state3_i3_end),
        (robot_state3_i4_start, robot_state3_i4_end),
    ]
)

# Комната 4
obstacle_coords4 = [
     (4, 9), (5, 9), (6, 9), (7, 9), (8, 9), (9, 9), (10, 9)
]

robot_state4_i1_start = RobotState(x=11, y=3, theta=0)
robot_state4_i1_end = RobotState(x=9, y=5, theta=0)

robot_state4_i2_start = RobotState(x=3, y=2, theta=0)
robot_state4_i2_end = RobotState(x=11, y=2, theta=0)

robot_state4_i3_start = RobotState(x=5, y=5, theta=math.radians(30))
robot_state4_i3_end = RobotState(x=5, y=5, theta=math.radians(60))

robot_state4_i4_start = RobotState(x=5, y=5, theta=math.radians(60))
robot_state4_i4_end = RobotState(x=5, y=5, theta=math.radians(30))


scenario4 = ScenarioSetup(
    obstacle_coords=obstacle_coords4,
    robot_states=[
        (robot_state4_i1_start, robot_state4_i1_end),
        (robot_state4_i2_start, robot_state4_i2_end),
        (robot_state4_i3_start, robot_state4_i3_end),
        (robot_state4_i4_start, robot_state4_i4_end),
    ]
)

def create_room(
        room_size: int,
        obstacle_positions: Iterable[Tuple[int, int]],
        robot_state: RobotState,
) -> Room:
    room = np.zeros((room_size, room_size))
    for obstacle_position in obstacle_positions:
        room[obstacle_position[::-1]] = 1

    for i in range(room_size):
        room[(0, i)] = 1
        room[(i, 0)] = 1
        room[(i, room_size - 1)] = 1
        room[(room_size - 1, i)] = 1

    room[(robot_state.x, robot_state.y)[::-1]] = 0.5

    return Room(map_as_2d_array=room, map_size=room_size, obstacle_positions=obstacle_positions)

def create_scan_result(state: RobotState, scan_distances, angles, room, obstacle_positions) -> None:
    """Создаёт карту, стены, робота и результаты сканирования"""
    plt.figure(figsize=(8, 8))
    plt.imshow(room, cmap='Greys', origin='lower')
    plt.scatter(*zip(*obstacle_positions), color='yellow', label='Blocks')
    plt.scatter(state.x, state.y, color='purple', label='Robot', s=100)

    angles = np.radians(angles)

    for i, (angle, distance) in enumerate(zip(angles, scan_distances)):
        if distance != -1:
            end_point = (state.x + distance * np.cos(angle), state.y + distance * np.sin(angle))
            color = 'red' if i == 0 else 'blue'
            linewidth = 2.0 if i == 0 else 0.5
            plt.plot([state.x, end_point[0]], [state.y, end_point[1]], color=color, linewidth=linewidth, linestyle='--')

    plt.xlabel("Ось X")
    plt.ylabel("Ось Y")
    plt.title('Map with robot and scan result')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.show()

def perform_radar_measurement(state: RobotState, room, angle_increment=SCAN_STEP, max_range=MAX_SCAN_RANGE) -> (np.ndarray, np.ndarray):
    """
    Выполняет радарное измерение (аналог LIDAR).
    Возвращает измеренные расстояния (-1 означает, что объект находится дальше максимальной дистанции) и углы.
    """
    start_degrees = math.degrees(state.theta)
    measurement_angles = np.arange(start_degrees, start_degrees + SCAN_ANGLE_RANGE, angle_increment) % 360
    measured_distances = []

    for measurement_angle in measurement_angles:
        radian_angle = np.radians(measurement_angle)
        detected_distance = -1
        for distance in range(1, max_range):
            coord_x = int(state.x + distance * np.cos(radian_angle))
            coord_y = int(state.y + distance * np.sin(radian_angle))

            if coord_x < 0 or coord_y < 0 or coord_x >= MAP_SIZE or coord_y >= MAP_SIZE or room[coord_y, coord_x] == 1:
                detected_distance = distance
                break

        measured_distances.append(detected_distance)

    return np.array(measured_distances), measurement_angles



def convert_scan_data_to_coordinates(distance_measurements, angle_measurements):
    """
    Конвертирует измерения дистанции и углов в координаты облака точек.
    Предполагается, что начальное положение сканирующего устройства находится в (0, 0) с углом 0 радиан.
    """
    radian_angles = np.radians(np.arange(0, SCAN_ANGLE_RANGE, SCAN_STEP))

    effective_distances = distance_measurements[distance_measurements > 0]
    effective_radian_angles = radian_angles[distance_measurements > 0]

    coordinates_x = effective_distances * np.cos(effective_radian_angles)
    coordinates_y = effective_distances * np.sin(effective_radian_angles)

    return np.column_stack((coordinates_x, coordinates_y))

def display_point_sets(point_set1, point_set2, modified_point_set=None):
    """
      Отображает два или три набора точек на графике.
      point_set1 и point_set2 - исходные наборы точек.
      modified_point_set1 - опционально, трансформированный набор точек point_set1.
      """
    plt.figure(figsize=(8, 8))

    plt.scatter(point_set1[:, 0], point_set1[:, 1], color='orange', marker='*', label='First point cloud')

    plt.scatter(point_set2[:, 0], point_set2[:, 1], color='green', marker='o', label='Second point cloud')

    if modified_point_set is not None:
        plt.scatter(modified_point_set[:, 0], modified_point_set[:, 1], color='red', marker='x',
                    label='Transformed point cloud 1')

    plt.title("Point cloud comparison")
    plt.xlabel("Ось X")
    plt.ylabel("Ось Y")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def compute_point_cloud_transformation(cloud_1, cloud_2) -> [np.array, Transformation]:
    """
    Расчет трансформации между двумя наборами точек с использованием
    метода Coherent Point Drift (CPD).

    Возвращает преобразованный набор точек и параметры преобразования.
    """
    point_drift_reg = RigidRegistration(X=cloud_2, Y=cloud_1)
    transformed_cloud, transform_params = point_drift_reg.register()
    scale, rotation_matrix, translation_vector = transform_params

    transformed_cloud_final = point_drift_reg.transform_point_cloud(Y=cloud_1)
    transform_data = Transformation(s=scale, R=rotation_matrix, t=translation_vector)

    return transformed_cloud_final, transform_data

def display_scan_results(start_state: RobotState, end_state: RobotState, calculated_transformation: Transformation) -> None:
    PRECISION = 3

    # Вычисление угла поворота
    calculated_cos_theta = np.round(calculated_transformation.R[0, 0], 5)  # округление для избежания ошибок в acos
    estimated_sign_of_theta = 1 if calculated_transformation.R[0, 1] < 0 else -1 if calculated_transformation.R[1, 0] <= 0 else 1

    # Рассчитываемая трансформация на основе данных сканирования
    estimated_theta = math.acos(calculated_cos_theta)
    estimated_theta *= estimated_sign_of_theta
    estimated_x = -calculated_transformation.t[0]
    estimated_y = -calculated_transformation.t[1]

    # Фактическая трансформация
    true_x_shift = end_state.x - start_state.x
    true_y_shift = end_state.y - start_state.y
    true_theta_shift = end_state.theta - start_state.theta

    # Разница между фактической и рассчитанной трансформацией
    delta_x = true_x_shift - estimated_x
    delta_y = true_y_shift - estimated_y
    delta_theta = true_theta_shift - estimated_theta

    # Округление для улучшения читаемости
    estimated_theta = np.round(estimated_theta, PRECISION)
    estimated_x = np.round(estimated_x, PRECISION)
    estimated_y = np.round(estimated_y, PRECISION)
    true_x_shift = np.round(true_x_shift, PRECISION)
    true_y_shift = np.round(true_y_shift, PRECISION)
    true_theta_shift = np.round(true_theta_shift, PRECISION)
    delta_x = np.round(abs(delta_x), PRECISION)
    delta_y = np.round(abs(delta_y), PRECISION)
    delta_theta = np.round(abs(delta_theta), PRECISION)

    delta_x_percentage = np.round(delta_x / MAP_SIZE * 100, PRECISION)
    delta_y_percentage = np.round(delta_y / MAP_SIZE * 100, PRECISION)
    delta_theta_percentage = np.round(delta_theta / math.radians(360) * 100, PRECISION)

    vector_length_difference = round(math.sqrt(delta_x ** 2 + delta_y ** 2), 3)
    diagonal_length = MAP_SIZE * math.sqrt(2)
    accuracy_percentage = round(vector_length_difference / diagonal_length * 100, 3)

    print(f"{MAP_SIZE=} {SCAN_STEP=} {start_state=} {end_state=}")
    print(f"Accuracy of X/Y transformation vector - {accuracy_percentage}%")

    print("True Transformation")
    print(f"\tX\t{true_x_shift}")
    print(f"\tY\t{true_y_shift}")
    print(f"\tθ(rad)\t{true_theta_shift}")

    print("Calculated Transformation")
    print(f"\tX\t{estimated_x}")
    print(f"\tY\t{estimated_y}")
    print(f"\tθ(rad)\t{estimated_theta}")

    print("Differences in Transformations (% of room size)")
    print(f"\tX\t{delta_x} ({delta_x_percentage}%)")
    print(f"\tY\t{delta_y} ({delta_y_percentage}%)")
    print(f"\tθ(rad)\t{delta_theta} ({delta_theta_percentage}%)")

def conduct_experiment(obstacle_positions, pose1, pose2):
    room = create_room(MAP_SIZE, obstacle_positions, pose1)
    scan_distances, angles = perform_radar_measurement(pose1, room.map_as_2d_array)
    scan_distances2, angles2 = perform_radar_measurement(pose2, room.map_as_2d_array)
    point_cloud1 = convert_scan_data_to_coordinates(scan_distances, angles)
    point_cloud2 = convert_scan_data_to_coordinates(scan_distances2, angles2)
    point_cloud1_transformed, transformation = compute_point_cloud_transformation(point_cloud1, point_cloud2)
    create_scan_result(pose1, scan_distances, angles, room.map_as_2d_array, room.obstacle_positions)
    create_scan_result(pose2, scan_distances2, angles2, room.map_as_2d_array, room.obstacle_positions)
    display_point_sets(point_cloud1, point_cloud2, point_cloud1_transformed)
    display_scan_results(pose1, pose2, transformation)

for i, (pose1, pose2) in enumerate(scenario1.robot_states, start=1):
    print(f"Room 1. Iteration {i}")
    conduct_experiment(scenario1.obstacle_coords, pose1, pose2)

for i, (pose1, pose2) in enumerate(scenario2.robot_states, start=1):
    print(f"Room 2. Iteration {i}")
    conduct_experiment(scenario2.obstacle_coords, pose1, pose2)

for i, (pose1, pose2) in enumerate(scenario3.robot_states, start=1):
    print(f"Room 3. Iteration {i}")
    conduct_experiment(scenario3.obstacle_coords, pose1, pose2)

for i, (pose1, pose2) in enumerate(scenario4.robot_states, start=1):
    print(f"Room 4. Iteration {i}")
    conduct_experiment(scenario4.obstacle_coords, pose1, pose2)

