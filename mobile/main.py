from mobile_simulator import *
from filters.dumb_filter import DumbFilter
from filters.moving_average_filter import MovingAverageFilter
from filters.kalman_filter import KalmanFilter
from filters.linear_displacement_filter import LinearDisplacementFilter
from filters.opencv_kalman_filter import OpencvKalmanFilter
from filters.opencv_kalman_filter_with_control import OpencvKalmanFilterWithControl

if __name__ == '__main__':
    mobile = MobileAPI(pos_measurement_variance=5,
                       acceleration_change_frequency=25,
                       measurement_frequency=8)

    filters = []
    # filters.append(DumbFilter())
    # filters.append(MovingAverageFilter(5))
    filters.append(LinearDisplacementFilter())
    # filters.append(KalmanFilter(2, 3))
    # filters.append(OpencvKalmanFilter())
    # filters.append(OpencvKalmanFilterWithControl(mobile.get_accelerometer_data))

    simulator = Simulator(mobile, filters)
    simulator.show_real_position = True
    while True:
        simulator.step()
