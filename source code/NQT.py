import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import hsv_to_rgb


def read_parameters(filename="init.txt"):
    params = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                if '=' in line:
                    parts = line.split('=', 1)
                    key = parts[0].strip()
                    value = parts[1].strip()
                    if '#' in value:
                        value = value.split('#')[0].strip()
                    if not value:
                        continue
                    if '.' in value:
                        params[key] = float(value)
                    else:
                        params[key] = int(value)
    return params

params = read_parameters()

space_cell = params['space_cell']
time_step = params['time_step']
particle_radius = params['particle_radius']
N_compact = params['N_compact']
size_compact_limit = params['size_compact_limit']
N_particles = params['N_particles']
box_size = params['box_size']
partition_position = params['partition_position']
min_speed = params['min_speed']
max_speed = params['max_speed']


class CompactSpaceSimulation:
    def __init__(self, N_compact, N_particles,
                 box_size, partition_position,
                 min_limit, max_limit,
                 dt, particle_radius):

        self.N_compact = N_compact
        self.N_particles = N_particles
        self.box_size = box_size
        self.partition_position = partition_position
        self.dt = dt
        self.particle_radius = particle_radius

        self.total_particles = 2 * N_particles
        self.positions = np.zeros((self.total_particles, 3 + N_compact))
        self.velocities = np.zeros((self.total_particles, 3 + N_compact))

        common_ax_limits = np.random.uniform(min_limit, max_limit, N_compact)
        self.ax_limits = np.tile(common_ax_limits, (self.total_particles, 1))

        self._initialize_particles()

        self.colors = self._generate_colors()

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

    def _initialize_particles(self):
        for i in range(self.N_particles):
            self.positions[i, 0] = np.random.uniform(-self.box_size / 2 + self.particle_radius,
                                                     self.partition_position - space_cell - self.particle_radius)
            self.positions[i, 1] = np.random.uniform(-self.box_size / 2 + self.particle_radius,
                                                     self.box_size / 2 - self.particle_radius)
            self.positions[i, 2] = np.random.uniform(-self.box_size / 2 + self.particle_radius,
                                                     self.box_size / 2 - self.particle_radius)

            for j in range(self.N_compact):
                self.positions[i, 3 + j] = np.random.uniform(0, self.ax_limits[i, j])

            speed = np.random.uniform(min_speed, max_speed)
            direction = np.random.randn(3 + self.N_compact)
            direction /= np.linalg.norm(direction)
            self.velocities[i] = speed * direction

        for i in range(self.N_particles, 2 * self.N_particles):
            self.positions[i, 0] = np.random.uniform(self.partition_position + space_cell + self.particle_radius,
                                                     self.box_size / 2 - self.particle_radius)
            self.positions[i, 1] = np.random.uniform(-self.box_size / 2 + self.particle_radius,
                                                     self.box_size / 2 - self.particle_radius)
            self.positions[i, 2] = np.random.uniform(-self.box_size / 2 + self.particle_radius,
                                                     self.box_size / 2 - self.particle_radius)

            for j in range(self.N_compact):
                self.positions[i, 3 + j] = np.random.uniform(0, self.ax_limits[i, j])

            speed = np.random.uniform(min_speed, max_speed)
            direction = np.random.randn(3 + self.N_compact)
            direction /= np.linalg.norm(direction)
            self.velocities[i] = speed * direction

    def _generate_colors(self):
        colors = []

        for i in range(self.N_particles):
            hue = np.random.uniform(0, 0.15)
            saturation = np.random.uniform(0.7, 1.0)
            value = np.random.uniform(0.7, 1.0)
            colors.append(hsv_to_rgb([hue, saturation, value]))

        for i in range(self.N_particles):
            hue = np.random.uniform(0.5, 0.8)
            saturation = np.random.uniform(0.7, 1.0)
            value = np.random.uniform(0.7, 1.0)
            colors.append(hsv_to_rgb([hue, saturation, value]))

        return np.array(colors)

    def _handle_compact_dimensions(self):
        for i in range(self.total_particles):
            for j in range(self.N_compact):
                compact_pos = self.positions[i, 3 + j]
                compact_vel = self.velocities[i, 3 + j]
                ax_limit = self.ax_limits[i, j]

                new_pos = compact_pos + compact_vel * self.dt

                if new_pos < 0:
                    new_pos = -new_pos
                    self.velocities[i, 3 + j] = -compact_vel
                elif new_pos > ax_limit:
                    new_pos = 2 * ax_limit - new_pos
                    self.velocities[i, 3 + j] = -compact_vel

                self.positions[i, 3 + j] = new_pos

    def _handle_3d_boundaries(self):
        half_size = self.box_size / 2

        for i in range(self.total_particles):
            current_radius = self.particle_radius

            for dim in range(3):
                pos = self.positions[i, dim]
                vel = self.velocities[i, dim]

                if abs(pos) + current_radius > half_size:
                    self.velocities[i, dim] = -vel
                    if pos > 0:
                        self.positions[i, dim] = half_size - current_radius
                    else:
                        self.positions[i, dim] = -half_size + current_radius

    def _handle_partition_collision(self):
        for i in range(self.total_particles):
            current_radius = self._get_visible_radius(i)
            if current_radius > 0.01:
                x_pos = self.positions[i, 0]
                if (abs(x_pos - self.partition_position) < space_cell + current_radius and
                        abs(self.velocities[i, 0]) > 0 and
                        np.sign(x_pos - self.partition_position) != np.sign(self.velocities[i, 0])):
                    self.velocities[i, 0] = -self.velocities[i, 0]
                    if x_pos > self.partition_position:
                        self.positions[i, 0] = self.partition_position + space_cell + current_radius
                    else:
                        self.positions[i, 0] = self.partition_position - space_cell - current_radius

    def _handle_particle_repulsion(self):
        for i in range(self.total_particles):
            for j in range(i + 1, self.total_particles):
                radius_i = self._get_visible_radius(i)
                radius_j = self._get_visible_radius(j)

                if radius_i < 0.01 and radius_j < 0.01:
                    continue

                delta = self.positions[i, :3] - self.positions[j, :3]
                distance = np.linalg.norm(delta)

                sum_radii = radius_i + radius_j

                if distance < sum_radii:
                    if distance > 1e-10:
                        direction = delta / distance
                    else:
                        direction = np.random.randn(3)
                        direction /= np.linalg.norm(direction)

                    overlap = sum_radii - distance
                    correction_i = 0.5 * overlap * direction
                    correction_j = -0.5 * overlap * direction

                    self.positions[i, :3] += correction_i
                    self.positions[j, :3] += correction_j

                    delta = self.positions[i, :3] - self.positions[j, :3]
                    distance = np.linalg.norm(delta)
                    if distance > 1e-10:
                        direction = delta / distance

                    relative_velocity = self.velocities[i, :3] - self.velocities[j, :3]
                    velocity_component = np.dot(relative_velocity, direction)

                    if velocity_component < 0:
                        impulse = velocity_component * direction
                        self.velocities[i, :3] -= impulse
                        self.velocities[j, :3] += impulse

    def _get_visibility_factor(self, particle_idx):
        compact_coords = self.positions[particle_idx, 3:]
        ax_limits = self.ax_limits[particle_idx]

        if np.all(ax_limits < 1e-10):
            return 1.0

        max_deviation = 0
        for coord, limit in zip(compact_coords, ax_limits):
            if limit < 1e-10:
                continue

            abs_dist = abs(coord - limit / 2)
            max_deviation = max(max_deviation, abs_dist)

        if max_deviation == 0:
            return 1.0

        visibility = max(0, 1.0 - max_deviation)

        if visibility < space_cell:
            return 0.0

        return visibility

    def _get_visible_radius(self, particle_idx):
        visibility = self._get_visibility_factor(particle_idx)
        return self.particle_radius * visibility

    def update(self):
        self.positions += self.velocities * self.dt

        self._handle_compact_dimensions()

        self._handle_3d_boundaries()

        self._handle_partition_collision()

        self._handle_particle_repulsion()

    def visualize(self):
        self.ax.clear()

        self._draw_3d_box()

        for i in range(self.total_particles):
            visibility = self._get_visibility_factor(i)
            if visibility > space_cell:
                pos = self.positions[i, :3]
                visible_radius = self._get_visible_radius(i)
                color = self.colors[i]

                u = np.linspace(0, 2 * np.pi, 10)
                v = np.linspace(0, np.pi, 10)
                x = pos[0] + visible_radius * np.outer(np.cos(u), np.sin(v))
                y = pos[1] + visible_radius * np.outer(np.sin(u), np.sin(v))
                z = pos[2] + visible_radius * np.outer(np.ones(np.size(u)), np.cos(v))

                alpha = 0.3 + 0.4 * visibility
                self.ax.plot_surface(x, y, z, color=color, alpha=alpha)

        self.ax.set_xlim([-self.box_size / 2, self.box_size / 2])
        self.ax.set_ylim([-self.box_size / 2, self.box_size / 2])
        self.ax.set_zlim([-self.box_size / 2, self.box_size / 2])
        self.ax.set_box_aspect([1, 1, 1])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

    def _draw_3d_box(self):
        half_size = self.box_size / 2
        partition_thickness = 2 * space_cell

        x_min = self.partition_position - partition_thickness / 2
        x_max = self.partition_position + partition_thickness / 2

        x = np.array([x_min, x_max])
        y = np.array([-half_size, half_size])
        z = np.array([-half_size, half_size])

        X, Y = np.meshgrid(x, y)
        Z, _ = np.meshgrid(z, x)

        color = 'darkgray'
        self.ax.plot_surface(X, Y, np.ones_like(X) * (-half_size), alpha=0.3, color=color)  # нижняя
        self.ax.plot_surface(X, Y, np.ones_like(X) * half_size, alpha=0.3, color=color)  # верхняя
        self.ax.plot_surface(X, np.ones_like(Y) * (-half_size), Z, alpha=0.3, color=color)  # задняя
        self.ax.plot_surface(X, np.ones_like(Y) * half_size, Z, alpha=0.3, color=color)  # передняя
        self.ax.plot_surface(np.ones_like(X) * x_min, Y, Z, alpha=0.3, color=color)  # левая
        self.ax.plot_surface(np.ones_like(X) * x_max, Y, Z, alpha=0.3, color=color)  # правая


def run_simulation():
    sim = CompactSpaceSimulation(
        N_compact=N_compact,
        N_particles=N_particles,
        box_size=box_size,
        partition_position=partition_position,
        min_limit=min(2*particle_radius, size_compact_limit),
        max_limit=size_compact_limit,
        dt=time_step,
        particle_radius=particle_radius
    )

    def animate(frame):
        sim.update()
        sim.visualize()
        return sim.ax

    ani = FuncAnimation(sim.fig, animate, frames=500, interval=50, blit=False)
    plt.show()

run_simulation()


