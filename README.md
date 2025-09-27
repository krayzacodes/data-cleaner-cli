# 🧹 Data Cleaner CLI

A lightweight yet powerful command-line tool to **clean and normalize messy CSV files**.  
Built with **Python 3, Pandas, Numpy**.

Author: **Mizgin Y.**  
Repo: [krayzacodes/data-cleaner-cli](https://github.com/krayzacodes/data-cleaner-cli)

---

## ✨ Features
- 🗑️ Remove duplicates  
- ✂️ Trim extra spaces  
- 🧩 Handle missing values (numeric & text)  
- 🔢 Convert numeric & date columns  
- 📧 Normalize email addresses  
- 🗂️ Select / drop / rename columns  
- 📊 Dataset summary (rows, columns, null counts)  
- 📝 Generate a **Markdown report** of changes  

---

## 📦 Installation

Clone the repository:
```bash
git clone https://github.com/krayzacodes/data-cleaner-cli.git
cd data-cleaner-cli
pip install -r requirements.txt

---

## 🚀 Usage

```bash
# Clean a CSV file and save results
python main.py samples/dirty.csv samples/clean.csv

# Dry-run (just preview changes without saving)
python main.py samples/dirty.csv --dry

# Select only specific columns
python main.py samples/dirty.csv samples/clean.csv --select name,email,salary

# Drop certain columns
python main.py samples/dirty.csv samples/clean.csv --drop notes

