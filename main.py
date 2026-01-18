# main.py
from data_gen import generate_street_light_dataset
from plots import create_visualizations
from ann_model import run_ann_pipeline
from ea_optimizer import run_ea_pipeline


def main():
    generate_street_light_dataset(output_path="street_light_dataset.csv")
    create_visualizations(csv_path="street_light_dataset.csv")
    run_ann_pipeline(csv_path="street_light_dataset.csv")
    run_ea_pipeline(csv_path="street_light_dataset.csv")


if __name__ == "__main__":
    main()
