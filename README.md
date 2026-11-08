# Generative 3D Cities

A pipeline that procedurally generates cities as data grids and turns them into colored, 3D-printable models in Blender. Built with the Michigan Data Science Team (MDST). **Status: in progress.** The generation and export pipeline works end to end, and you can now describe a city in plain English and have a trained model turn the sentence into a generated city (`text2city/infer.py`).

<img width="1200" height="800" alt="image" src="https://github.com/user-attachments/assets/cd4ec8e7-afa3-4948-a106-228dc6bbcdba" />

## Overview

Every city is represented as a fixed-size grid (128×128) with four channels:

| Channel | Type | Meaning |
|---|---|---|
| `terrain_height` | float32 | Ground elevation |
| `is_water` | bool | Lakes, rivers, harbors |
| `building_height` | float32 | Building height (0 = no building) |
| `land_use` | uint8 | Building / road / green space / empty ground |

The grid is saved as a `.npz` file, which every stage of the pipeline reads or writes.

<img width="2700" height="750" alt="image" src="https://github.com/user-attachments/assets/a2b4b1a7-129b-4a21-a861-9fcbd606d795" />

## Pipeline

```
generate_city.py  ->  mycity.npz  ->  visualize_2d.py  ->  mycity_preview.png
                          |
                          +-> export_to_blender.py (runs inside Blender) -> mycity.blend + mycity.stl

"a dense city in the hills..."  ->  text2city model  ->  10 values  ->  generate_city.py command  ->  .npz
                                    (Week 5)                            (Week 6: infer.py)
```

1. **Terrain generation** (`citygen/terrain.py`): fractal noise shaped by an archetype (flat, hills), with water marked below the water level.
2. **Building generation** (`citygen/buildings.py`): cluster centers, urban intensity falloff, no buildings on water or steep slopes, then roads with gaps and green space.
3. **2D preview** (`scripts/visualize_2d.py`): four panels for terrain, water, building height, and land use.
4. **Blender export** (`scripts/export_to_blender.py`): builds terrain and building meshes, colors them by land use and height, and saves a `.blend` and a printable `.stl`.
5. **Text-to-parameters** (`text2city/`): a trained model reads a sentence and predicts the generator's settings (see below).
6. **Sentence to city** (`text2city/infer.py`): turns the prediction into a `generate_city.py` command, prints it, and runs it.

## Quick start

Run all commands from the repo root.

```bash
# 1. Set up
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Generate a city
python scripts/generate_city.py --size 128 --terrain hills --city polycentric \
  --clusters 3 --density 0.6 --cluster-tightness 0.6 \
  --green-amount 0.25 --road-continuity 0.7 --seed 42 --out data/mycity.npz

# 3. Preview in 2D
python scripts/visualize_2d.py data/mycity.npz --out data/mycity_preview.png

# 4. Export to Blender (requires Blender 3.x or 4.x)
blender --background --python scripts/export_to_blender.py -- data/mycity.npz

# 5. Open the model (switch viewport shading to Material Preview)
blender data/mycity.blend
```

On macOS, Blender is not on your PATH by default. Add an alias:

```bash
echo 'alias blender="/Applications/Blender.app/Contents/MacOS/Blender"' >> ~/.zshrc
source ~/.zshrc
```

### Generate a city from a sentence

This needs a trained model first (see [Text-to-parameter model](#text-to-parameter-model-week-5)).

```bash
python text2city/infer.py "a dense city in the hills with three business hubs" --preview
```

It prints the 10 predicted values and the exact `generate_city.py` command they turn into, runs it, and saves `data/from_text.npz` plus a 2D preview. Blender export is the same step as above, on `data/from_text.npz`.

## Generator options

| Flag | What it controls |
|---|---|
| `--size` | Grid size (default 128) |
| `--terrain` | Terrain archetype (`flat`, `hills`) |
| `--city` | City archetype (e.g. `polycentric`, `linear`) |
| `--clusters` | Number of city centers |
| `--density` | Overall building density |
| `--cluster-tightness` | How tightly buildings pack around centers |
| `--green-amount` | Share of land used for green space |
| `--road-continuity` | How unbroken the road network is |
| `--seed` | Random seed for reproducible cities |

## Text-to-parameter model (Week 5)

A small model that reads a plain-English description and predicts the generator's 10 parameters: terrain, city type, number of clusters, density, cluster tightness, relief, water level, green amount, green override, and road continuity. Parameters the sentence doesn't mention keep their defaults.

**How it works**

- **Synthetic data** (`generate_dataset.py`): builds 15,000 sentences from randomized parameter settings. Each sentence mentions only some of the parameters, the way a real person would, and the rest stay at their defaults.
- **Targets** (`dataset.py`): turns the CSV into training targets: class indices for categorical parameters, 0–1 rescaled values for continuous ones, and 0/1 flags for binary ones.
- **Model** (`model.py`): a frozen sentence embedding (`all-MiniLM-L6-v2`) feeds a small shared network with one output head per parameter.
- **Training** (`train.py`): cross-entropy for categorical heads, mean squared error for continuous heads, and binary cross-entropy for the binary head.
- **Evaluation** (`evaluate.py`): two tests. The first uses held-out synthetic sentences. The second uses hand-written sentences with different wording, to check that the model learned meaning and didn't just memorize the phrase templates.
- **Stale-checkpoint guard** (`freshness.py`): the checkpoint stores a fingerprint of the dataset, the model code, and the terrain and city choices from `citygen`. If any of them change after training, `evaluate.py` and `infer.py` warn that the checkpoint is stale.

**Build the model** (from the repo root)

```bash
python text2city/generate_dataset.py --n 15000 --seed 0
python text2city/train.py --embedder sentence-transformer
python text2city/evaluate.py
```

Training writes the model to `text2city/checkpoint/` (`model.pt` and `embedder.txt`). `data/` and `text2city/checkpoint/` are not tracked in git, so everyone rebuilds them with the commands above.

**What we found**

- Held-out synthetic sentences score close to perfect, since they come from the same phrase banks the model trained on.
- The hand-written sentences are the real test. The misses show where the model only learned the training vocabulary. The fix is to add phrases to the banks in `generate_dataset.py`, then regenerate and retrain. A bigger dataset from the same banks would not help.

## Sentence to city (Week 6)

`text2city/infer.py` connects the model to the generator.

```bash
python text2city/infer.py "a quiet town on flat land" --dry-run      # print the command only
python text2city/infer.py "a dense city in the hills with three business hubs" --seed 7 --preview
```

| Flag | What it does |
|---|---|
| `--seed` | Same seed and same sentence give the same city (a random one is picked and printed if you leave it out) |
| `--size` | Grid size (default 128) |
| `--out` | Where to save the `.npz` (default `data/from_text.npz`) |
| `--preview` | Also save a 2D PNG next to the `.npz` |
| `--dry-run` | Print the command and stop |
| `--allow-stale` | Use a stale checkpoint anyway |

**Design: the command line is the hand-off.** `infer.py` never calls the generator directly. It predicts the 10 values, converts them to keyword arguments with `to_generator_kwargs()` in `schema.py`, converts those to flags with `kwargs_to_cli_args()`, and prints the resulting `generate_city.py` command as a receipt before running it. If a city looks wrong, copy the printed command, change a number, and run it yourself. That shows which half is wrong: if the predicted numbers are wrong, the model is to blame; if the numbers are right but the city isn't, the generator is.

**Does the city match the sentence?** Week 5 checked whether the model predicts the right numbers. Week 6 checks the city itself. The notebook measures each generated city (water share, plus building, road and green share of land cells) and compares pairs of sentences that should differ, such as "extremely dense" against "sparse", averaged over three seeds. Each miss is labeled as a model miss or a generator miss. One of the pairs is my own.

## Project structure

```
citygen/
  city_grid.py      # CityGrid: the four-array data contract, save/load/stats
  terrain.py        # generate_terrain(...)
  buildings.py      # generate_buildings(...)
text2city/
  schema.py             # the 10 parameters, and to_generator_kwargs()
  embedders.py          # sentence embedding (and TF-IDF fallback)
  freshness.py          # stale-checkpoint check
  generate_dataset.py   # synthetic sentence + parameter dataset
  dataset.py            # CSV -> training targets
  model.py              # shared trunk + one head per parameter
  train.py              # training loop
  evaluate.py           # synthetic and hand-written tests
  infer.py              # sentence -> generate_city.py command -> city
  checkpoint/           # trained model (not tracked in git)
scripts/
  generate_city.py      # CLI: terrain -> buildings -> .npz
  visualize_2d.py       # 2D preview
  export_to_blender.py  # runs inside Blender: .npz -> .blend + .stl
notebooks/
  ...                   # weekly notebooks
docs/
  blender_export.md     # how a .npz becomes a .blend
```

`export_to_blender.py` runs in Blender's own Python, which has numpy but not scipy. That is why `citygen/__init__.py` only imports terrain and buildings when scipy is available.

## Roadmap

- [x] Procedural terrain and building generators
- [x] `.npz` city format (`CityGrid`)
- [x] 2D preview
- [x] Blender export with colored materials and printable STL
- [x] Road-continuity metric
- [x] Synthetic training dataset (thousands of randomized cities)
- [x] Text-to-parameter model (natural-language description to generator parameters)
- [x] Sentence to city: `infer.py` runs the model and the generator end to end
- [ ] Two-stage conditioned ML model: a terrain model feeds a building model so water and slopes suppress buildings coherently
- [ ] Visual interface (a small web page in front of `infer.py`)
- [ ] Final expo demo

## Tech stack

Python · NumPy · SciPy · Matplotlib · PyTorch · scikit-learn · sentence-transformers · Blender (bpy) · Jupyter · Git/GitHub
*Planned: Streamlit (interface)*

## Troubleshooting

| Error | Fix |
|---|---|
| `No module named 'citygen'` | Run from the repo root |
| `No module named 'numpy'` / `scipy` | Activate the venv and run `pip install -r requirements.txt` |
| `No module named 'scipy'` inside Blender | Use the fixed `citygen/__init__.py`; don't install scipy into Blender |
| Blender quits without saving | Add `--` before the `.npz` path |
| Model looks all grey | Switch viewport shading to Material Preview |
| `No module named 'torch'` / `sentence_transformers` | Activate the venv and run `pip install torch scikit-learn sentence-transformers` |
| `No checkpoint at ...` from `infer.py` | Train one: `python text2city/generate_dataset.py` then `python text2city/train.py` |
| STALE warning from `evaluate.py` or `infer.py` | The dataset, model code, or terrain list changed after training. Re-run `generate_dataset.py` and `train.py` |

## Credits
Built with the Michigan Data Science Team (MDST), University of Michigan.