import sys
import time
from datetime import datetime
from pathlib import Path

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLineEdit, QPushButton, QFormLayout, QSpinBox, \
    QDoubleSpinBox, QComboBox, QLabel

from file_writer import save
from genetic_algorithm import GeneticAlgorithm
from objective_function import hypersphere
from plotter import plot_function_3d, plot_population_3d, plot_best


class GeneticAlgorithmGUI(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Genetic Algorithm GUI')
        self.setGeometry(50, 80, 600, 600)

        layout = QVBoxLayout()

        form_layout = QFormLayout()

        self.population_size_input = QSpinBox()
        self.population_size_input.setRange(1, 1000)
        self.population_size_input.setValue(100)
        form_layout.addRow('Population Size:', self.population_size_input)

        self.n_generations_input = QSpinBox()
        self.n_generations_input.setRange(1, 1000)
        self.n_generations_input.setValue(100)
        form_layout.addRow('Number of Generations:', self.n_generations_input)

        self.bounds_input = QLineEdit()
        self.bounds_input.setText('-5, 5')
        form_layout.addRow('Bounds (comma separated):', self.bounds_input)

        self.N_input = QSpinBox()
        self.N_input.setRange(1, 10)
        self.N_input.setValue(2)
        form_layout.addRow('N:', self.N_input)

        self.selection_method_input = QComboBox()
        self.selection_methods = {
            'best': 'Best',
            'roulette': 'Roulette',
            'tournament': 'Tournament'
        }
        for value, display_text in self.selection_methods.items():
            self.selection_method_input.addItem(display_text, value)
        self.selection_method_input.setCurrentIndex(0)
        self.selection_method_input.currentIndexChanged.connect(self.update_tournament_size_visibility)
        form_layout.addRow('Selection Method:', self.selection_method_input)

        self.selection_ratio_input = QDoubleSpinBox()
        self.selection_ratio_input.setRange(0.0, 1.0)
        self.selection_ratio_input.setValue(0.3)
        form_layout.addRow('Selection Ratio:', self.selection_ratio_input)

        self.tournament_size_input = QSpinBox()
        self.tournament_size_input.setRange(1, 10)
        self.tournament_size_input.setValue(3)
        form_layout.addRow('Tournament Size:', self.tournament_size_input)

        self.mutation_type_input = QComboBox()
        self.mutation_types = {
            'gauss': 'Gauss',
            'uniform': 'Uniform'
        }
        for value, display_text in self.mutation_types.items():
            self.mutation_type_input.addItem(display_text, value)
        self.mutation_type_input.setCurrentIndex(0)
        form_layout.addRow('Mutation Type:', self.mutation_type_input)

        self.mutation_probability_input = QDoubleSpinBox()
        self.mutation_probability_input.setRange(0.0, 1.0)
        self.mutation_probability_input.setValue(0.7)
        form_layout.addRow('Mutation Probability:', self.mutation_probability_input)

        self.n_mutation_points_input = QSpinBox()
        self.n_mutation_points_input.setRange(1, 10)
        self.n_mutation_points_input.setValue(1)
        form_layout.addRow('Number of Mutation Points:', self.n_mutation_points_input)

        self.sigma_input = QDoubleSpinBox()
        self.sigma_input.setRange(-10.0, 10.0)
        self.sigma_input.setValue(1)
        form_layout.addRow('Sigma:', self.sigma_input)

        self.n_elites_input = QSpinBox()
        self.n_elites_input.setRange(0, 100)
        self.n_elites_input.setValue(1)
        form_layout.addRow('Number of Elites:', self.n_elites_input)

        self.cross_method_input = QComboBox()
        self.cross_methods = {
            'arithmetic': 'Arithmetic',
            'linear': 'Linear',
            'blend_alpha': 'Blend alpha',
            'blend_alpha_beta': 'Blend alpha beta',
            'average': 'Average'
        }
        for value, display_text in self.cross_methods.items():
            self.cross_method_input.addItem(display_text, value)
        self.cross_method_input.setCurrentIndex(0)
        self.cross_method_input.currentIndexChanged.connect(self.update_cross_visibility)
        form_layout.addRow('Cross Method:', self.cross_method_input)

        self.alpha_input = QDoubleSpinBox()
        self.alpha_input.setRange(-100.0, 100.0)
        self.alpha_input.setValue(0.25)
        form_layout.addRow('Alpha:', self.alpha_input)

        self.beta_input = QDoubleSpinBox()
        self.beta_input.setRange(-100.0, 100.0)
        self.beta_input.setValue(0.25)
        form_layout.addRow('Beta:', self.beta_input)

        self.p_inversion_input = QDoubleSpinBox()
        self.p_inversion_input.setRange(0.0, 1.0)
        self.p_inversion_input.setValue(0.0)
        form_layout.addRow('Inversion Probability:', self.p_inversion_input)

        layout.addLayout(form_layout)

        self.run_button = QPushButton('Run Genetic Algorithm')
        self.run_button.clicked.connect(self.run_genetic_algorithm)
        layout.addWidget(self.run_button)

        self.best_solution_label = QLabel('Best Solution:')
        layout.addWidget(self.best_solution_label)

        self.best_value_label = QLabel('Best Value:')
        layout.addWidget(self.best_value_label)

        self.global_best_solution_label = QLabel('Global best solution:')
        layout.addWidget(self.best_value_label)

        self.best_generation_label = QLabel('Best Generation:')
        layout.addWidget(self.best_generation_label)

        self.calculation_time = QLabel('Calculation Time:')
        layout.addWidget(self.calculation_time)

        self.setLayout(layout)
        self.update_tournament_size_visibility()
        self.update_cross_visibility()

        self.apply_styles()

    def apply_styles(self):
        self.setStyleSheet("""
            QWidget {
                font-family: Arial;
                font-size: 14px;
                background-color: #222;
                color: white;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                padding: 5px;
                margin: 5px;
                border: 2px solid #555;
                border-radius: 10px;
                background-color: #333;
                color: white;
            }
            QPushButton {
                padding: 5px;
                margin: 5px;
                background-color: #007BFF;
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QLabel {
                font-weight: bold;
                margin: 5px;
                font-size: 14px;
            }
        """)

    def update_tournament_size_visibility(self):
        selection_method = self.selection_method_input.currentData()
        if selection_method == 'tournament':
            self.tournament_size_input.show()
        else:
            self.tournament_size_input.hide()

    # def update_mutation_points_visibility(self):
    #     mutation_type = self.mutation_type_input.currentData()
    #     if mutation_type == 'n_points':
    #         self.n_mutation_points_input.show()
    #     else:
    #         self.n_mutation_points_input.hide()

    def update_cross_visibility(self):
        cross_method = self.cross_method_input.currentData()
        if cross_method == 'linear' or cross_method == 'average':
            self.alpha_input.hide()
        else:
            self.alpha_input.show()

        if cross_method == 'blend_alpha_beta':
            self.beta_input.show()
        else:
            self.beta_input.hide()

    def run_genetic_algorithm(self):
        self.run_button.setEnabled(False)
        self.run_button.setText('Calculating...')
        QApplication.processEvents()

        population_size = self.population_size_input.value()
        n_generations = self.n_generations_input.value()
        bounds = tuple(map(float, self.bounds_input.text().split(',')))
        N = self.N_input.value()

        selection_method = self.selection_method_input.currentData()
        selection_ratio = self.selection_ratio_input.value()
        tournament_size = self.tournament_size_input.value()
        mutation_type = self.mutation_type_input.currentData()
        mutation_probability = self.mutation_probability_input.value()
        n_mutation_points = self.n_mutation_points_input.value()
        n_elites = self.n_elites_input.value()
        cross_method = self.cross_method_input.currentData()
        alpha = self.alpha_input.value()
        beta = self.beta_input.value()
        sigma = self.sigma_input.value()
        p_inversion = self.p_inversion_input.value()

        ga = GeneticAlgorithm(
            objective_function=hypersphere,
            population_size=population_size,
            n_generations=n_generations,
            bounds=bounds,
            N=N,
            selection_method=selection_method,
            selection_ratio=selection_ratio,
            tournament_size=tournament_size,
            mutation_method=mutation_type,
            p_mutation=mutation_probability,
            n_mutation_points=n_mutation_points,
            sigma=sigma,
            n_elites=n_elites,
            cross_method=cross_method,
            p_inversion=p_inversion,
            alpha=alpha,
            beta=beta,
        )

        start = time.time()
        result = ga.evolve()
        end = time.time()

        result['time'] = end - start

        global_optimum_solution = [0] * N
        global_optimum_value = 0

        self.best_solution_label.setText(f'Best Solution:\n{result["best_solution"]}')
        self.best_value_label.setText(f'Best Value:\n{result["best_value"]}')
        self.global_best_solution_label.setText(f'Global Optimum Solution:\n{global_optimum_solution}')
        self.best_generation_label.setText(
            f'Global Optimum Value:\n{global_optimum_value}\nBest Generation:\n{result["best_generation"]}')
        self.calculation_time.setText(f"Calculation Time: {end - start:.4f} seconds")

        dir_name = f"../results/{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}"
        Path(dir_name).mkdir(parents=True, exist_ok=True)
        save(f"{dir_name}/wynik.txt", result)
        plot_function_3d(hypersphere, f"{dir_name}/funkcja_celu.png")

        for population in ga.population_history:
            plot_population_3d(population[1], population[2], f"Generacja nr. {population[0]}",
                               f"{dir_name}/generacja-nr-{population[0]}.png")

        plot_best(ga.best_values_history, "Wykres wartości najlepszego rozwiązania w zależności od generacji",
                  f"{dir_name}/wykres.png")

        self.run_button.setEnabled(True)
        self.run_button.setText('Run Genetic Algorithm')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = GeneticAlgorithmGUI()
    ex.show()
    sys.exit(app.exec_())