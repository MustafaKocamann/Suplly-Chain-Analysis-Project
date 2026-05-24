"""
Data preparation pipeline for supply-chain delivery performance modeling.

This script prepares a modeling table for two binary classification targets:
    1. late_delivery
    2. fraud

It is designed to be called from the EDA notebook after `df` is loaded and
time-based features such as `order_year`, `order_month`, `order_week_day`,
`order_hour`, and `order_quarter` have been created.
"""

from __future__ import annotations

import json
import joblib
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.preprocessing import LabelEncoder


OUTPUT_DATA_PATH = Path("train_data_prepared.csv")
OUTPUT_ENCODING_PATH = Path("encoding_maps.json")

LABEL_ENCODE_COLUMNS = [
    'Type',
    'Customer Segment',
    'Market',
    'Customer Country',
    'Order Country',
    'Department Name',
    'Category Name',
    'Order Region',
    'Customer State',
    'Order State',
    'Shipping Mode',
]


def print_section(title: str) -> None:
    """Print a consistent section header for notebook/script auditability."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_class_balance(data: pd.DataFrame, target_col: str) -> None:
    """Print target counts and percentages for class imbalance review."""
    counts = data[target_col].value_counts(dropna=False).sort_index()
    percentages = data[target_col].value_counts(
        normalize=True, dropna=False
    ).sort_index() * 100

    balance = pd.DataFrame(
        {
            "count": counts,
            "class_balance_pct": percentages.round(2),
        }
    )

    print(f"\n{target_col} value counts and class balance:")
    print(balance)


def safe_drop(data: pd.DataFrame, columns: list[str], group_name: str) -> None:
    """
    Drop columns defensively.

    Missing columns are reported, not treated as fatal, because notebook state can
    vary slightly across runs after feature engineering.
    """
    print(f"\n{group_name}")
    for column in columns:
        try:
            if column in data.columns:
                data.drop(columns=column, inplace=True)
                print(f"  dropped: {column}")
            else:
                print(f"  skipped missing column: {column}")
        except Exception as exc:  # pragma: no cover - defensive audit branch
            print(f"  could not drop {column}: {exc}")


def create_training_copy(df: pd.DataFrame) -> pd.DataFrame:
    """Step 1: Create an isolated training copy so source EDA data is unchanged."""
    print_section("STEP 1 - CREATE TRAINING COPY")
    train_data = df.copy(deep=True)
    print(f"Training copy created. Shape: {train_data.shape}")
    return train_data


def create_binary_targets(train_data: pd.DataFrame) -> None:
    """Step 2: Create binary target columns before leakage columns are removed."""
    print_section("STEP 2 - CREATE BINARY TARGET COLUMNS")

    required_columns = ["Order Status", "Delivery Status"]
    missing = [col for col in required_columns if col not in train_data.columns]
    if missing:
        raise KeyError(
            "Cannot create targets because required columns are missing: "
            f"{missing}"
        )

    # KIYMETLİ VERİ BİLİMCİ NOTU: İki adet bağımsız ikili (binary) hedef değişkeni oluşturuyoruz.
    # 1. fraud: Siparişin şüpheli dolandırıcılık ('SUSPECTED_FRAUD') durumunda olup olmadığını belirtir.
    #    Bu hedef son derece dengesizdir (yaklaşık %2.3 pozitif sınıf). SMOTE gibi dengesiz sınıf metotları gerektirir.
    # 2. late_delivery: Teslimat durumunun gecikmiş ('Late delivery') olup olmadığını belirtir.
    #    Bu hedef ise dengelidir (yaklaşık %55 pozitif sınıf), özel dengesiz sınıf yönetimi gerektirmez.
    train_data["fraud"] = (
        train_data["Order Status"].eq("SUSPECTED_FRAUD").astype("int8")
    )
    train_data["late_delivery"] = (
        train_data["Delivery Status"].eq("Late delivery").astype("int8")
    )

    print_class_balance(train_data, "fraud")
    print_class_balance(train_data, "late_delivery")


def drop_modeling_exclusions(train_data: pd.DataFrame) -> None:
    """Step 3: Remove leakage, post-event, PII, redundant, and granular fields."""
    print_section("STEP 3 - DROP MODELING EXCLUSIONS")

    # VERİ SIZINTISI (DATA LEAKAGE) ÖNLEME:
    # 'Delivery Status', 'Late_delivery_risk' ve 'Order Status' kolonları, doğrudan tahmin etmek
    # istediğimiz hedef değişkenlerin ('late_delivery' ve 'fraud') türevleridir veya doğrudan kendileridir.
    # Bu kolonlar modelde kalırsa, model yapay bir şekilde %100 doğruluk gösterecek ancak gerçek hayatta (inference anında)
    # bu bilgi elimizde olmayacağı için model tamamen başarısız olacaktır. Bu yüzden kesinlikle kaldırılmalıdır.
    safe_drop(
        train_data,
        ["Delivery Status", "Late_delivery_risk", "Order Status"],
        "DATA LEAKAGE",
    )

    # EVENT-SONRASI (POST-EVENT) VERİLER:
    # Tahminleme işlemi siparişin verildiği "sipariş anında" (order time) yapılacaktır. Ancak kargonun
    # gönderilme tarihi olan 'Shipping date (DateOrders)', sipariş oluştuktan sonra kargolama anında belirlenir.
    # Modelin prediction-time gerçekliğine uyması ve geleceği tahmin etme yanılsamasına düşmemesi için çıkarılır.
    safe_drop(
        train_data,
        ["Shipping date (DateOrders)", "shipping date (DateOrders)"],
        "POST-EVENT (NOT AVAILABLE AT PREDICTION TIME)",
    )

    # HAM ZAMAN DAMGALARI (DATETIME RAW):
    # 'order date (DateOrders)' ve 'order_month_year' gibi ham tarih damgaları, doğrusal veya ağaç tabanlı modeller
    # tarafından doğrudan anlamlandırılamaz. Bunun yerine EDA aşamasında çıkarılmış olan 'order_year', 'order_month',
    # 'order_week_day', 'order_hour' gibi sayısal takvim özellikleri (decomposed features) modelde tutulur.
    safe_drop(
        train_data,
        ["order date (DateOrders)", "order_month_year"],
        "DATETIME RAW (DECOMPOSED FEATURES ALREADY EXIST)",
    )

    # KİŞİSEL VERİLERİN KORUNMASI (PII / NON-PREDICTIVE):
    # E-posta şifreleri, müşteri adları ve açık sokak adresleri gibi kişisel veriler, genel eğilimleri öğrenmek
    # yerine modelin tek tek satırları ezberlemesine (overfitting) yol açar. Ayrıca PII veri güvenliği kurallarına aykırıdır.
    safe_drop(
        train_data,
        [
            "Customer Email",
            "Customer Password",
            "Customer Fname",
            "Customer Lname",
            "Customer Full Name",
            "Customer Street",
        ],
        "PII / NON-PREDICTIVE",
    )

    # YÜKSEK KARDİNALİTELİ KİMLİK NUMARALARI (TRANSACTIONAL IDs):
    # Sipariş kimliği, müşteri kimliği ve ürün kart kimliği gibi benzersiz veya çok yüksek kardinaliteli kimlik alanları
    # modelin dallanma kriterlerinde (decision splits) her satırı ezbere ayırmasına neden olur ve tahmin gücü yoktur.
    safe_drop(
        train_data,
        [
            "Order Id",
            "Order Item Id",
            "Order Customer Id",
            "Product Card Id",
            "Order Item Cardprod Id",
            "Product Category Id",
        ],
        "HIGH CARDINALITY - NO PREDICTIVE VALUE",
    )

    # AŞIRI DETAYLI COĞRAFİ KONUMLAR VE METİNLER (OVERFITTING ENGELLEME):
    # Şehir isimleri ve posta kodları çok yüksek sayıda eşsiz sınıfa sahiptir. Posta kodlarının büyük kısmı eksiktir.
    # Bu alanlar gürültü (noise) ekler ve overfitting yaratır. 'Product Name' ise 'Category Name' ile zaten temsil edilmektedir.
    safe_drop(
        train_data,
        [
            "Order City",
            "Customer City",
            "Customer Zipcode",
            "Order Zipcode",
            "Product Name",
        ],
        "HIGH CARDINALITY LOCATION - TOO GRANULAR",
    )

    # ENLEM VE BOYLAM (GEO COORDINATES):
    # Müşteri ve sipariş koordinatları çok hassas noktasal verilerdir ve ağaç modellerinin ezberlemesine (overfitting)
    # yol açabilir. Projede bölgesel eğilimler 'Order Region' veya 'Order Country' gibi iş seviyesinde daha kararlı
    # ve makro coğrafi alanlarla zaten başarıyla temsil edilmektedir.
    safe_drop(
        train_data,
        ["Latitude", "Longitude"],
        "GEO COORDINATES - COVERED BY REGION/COUNTRY",
    )

    # REKABETÇİ / ÇOK DETAYLI GÖRSEL VE METİNLER:
    # Ürün açıklamaları ve ürün görsel linkleri serbest metindir ve kategori bilgisi ile zaten gruplanmıştır.
    safe_drop(
        train_data,
        ["Product Description", "Product Image"],
        "REDUNDANT WITH CATEGORY NAME",
    )

    # SIFIR VARYANS / SABİT KOLONLAR:
    # 'Product Status' kolonu tüm satırlar için 0 değerini almaktadır. Hiçbir varyansı olmayan (bilgi taşımayan)
    # kolonlar modelin karar eşiklerinde hiçbir ayrım gücüne sahip olamayacağı için veri setinden atılır.
    safe_drop(
        train_data,
        ["Product Status"],
        "CONSTANT / NO VARIANCE",
    )

    print(f"\nShape after drops: {train_data.shape}")
    print("\nRemaining columns:")
    print(train_data.columns.tolist())


def encode_with_label_encoders(
    train_data: pd.DataFrame, encoding_artifacts: dict[str, Any]
) -> dict[str, LabelEncoder]:
    """Step 4: Apply LabelEncoder to categorical columns for tree-based models."""
    print("\nLABEL ENCODING")

    # KIYMETLİ VERİ BİLİMCİ NOTU: Ağaç tabanlı modeller (Random Forest, XGBoost, LightGBM vb.) için 
    # Label Encoding en etkili ve verimli yöntemlerden biridir. Sütunların kardinalitesi (eşsiz değer sayısı) 
    # yüksek olduğunda One-Hot Encoding uygulanması, binlerce seyrek (sparse) sütun üreterek belleğin (RAM) 
    # taşmasına ve 'boyutun laneti' (curse of dimensionality) problemine yol açar. Ağaç algoritmaları sayısal 
    # büyüklük ilişkisini kendisi dallandırarak çözebildiği için Label Encoding burada en optimal tercihtir.
    label_encoders: dict[str, LabelEncoder] = {}

    for col in LABEL_ENCODE_COLUMNS:
        if col not in train_data.columns:
            print(f"  skipped missing: {col}")
            continue
        try:
            le = LabelEncoder()
            train_data[col] = le.fit_transform(train_data[col].astype(str))
            label_encoders[col] = le
            sample_mapping = dict(zip(le.classes_[:5], range(5)))
            print(
                f"  encoded: {col} | "
                f"classes: {len(le.classes_)} | "
                f"sample: {sample_mapping}"
            )
        except Exception as exc:
            print(f"  failed: {col} — {exc}")

    # Dağıtık sistemlerde ve inference (tahminleme) anında yeni veriyi aynı haritalama ile kodlamak
    # amacıyla fit edilmiş LabelEncoder sınıflarını diskte saklamak üzere encoding_artifacts yapısına kaydediyoruz.
    encoding_artifacts["label_encoding_maps"] = {
        col: {str(cls): int(idx) for idx, cls in enumerate(le.classes_)}
        for col, le in label_encoders.items()
    }

    return label_encoders


def one_hot_encode_group_a(
    train_data: pd.DataFrame, encoding_artifacts: dict[str, Any]
) -> None:
    """Step 4A: One-hot encode configured nominal categorical columns."""
    print("\nGROUP A - ONE-HOT ENCODING")

    # KIYMETLİ VERİ BİLİMCİ NOTU: Düşük kardinaliteli nominal (sıralanamayan) kategorik değişkenler için
    # One-Hot Encoding (kukla değişken oluşturma) doğrusal modeller (Logistic Regression vb.) için şarttır.
    # Burada drop_first=True parametresi kullanılarak ilk kolon düşürülür; bu sayede doğrusal modellerdeki
    # eşdoğrusallık (multicollinearity) ve 'dummy variable trap' (kukla değişken tuzağı) önlenmiş olur.
    one_hot_columns = [
        "Type",
        "Customer Segment",
        "Market",
        "Customer Country",
        "Order Country",
        "Department Name",
    ]

    available_columns = [col for col in one_hot_columns if col in train_data.columns]
    missing_columns = [col for col in one_hot_columns if col not in train_data.columns]

    for column in missing_columns:
        print(f"  skipped missing column: {column}")

    if not available_columns:
        encoding_artifacts["one_hot_columns"] = {}
        print("  no Group A columns available for one-hot encoding.")
        return

    cardinalities = {
        column: int(train_data[column].nunique(dropna=False))
        for column in available_columns
    }
    print("  cardinalities:", cardinalities)

    before_columns = set(train_data.columns)
    encoded = pd.get_dummies(
        train_data,
        columns=available_columns,
        drop_first=True,
        dtype="int8",
    )

    train_data.drop(columns=train_data.columns, inplace=True)
    for column in encoded.columns:
        train_data[column] = encoded[column]

    added_columns = sorted(set(train_data.columns) - before_columns)
    encoding_artifacts["one_hot_columns"] = {
        "source_columns": available_columns,
        "created_columns": added_columns,
        "drop_first": True,
    }

    print(f"  one-hot encoded columns: {available_columns}")
    print(f"  dummy columns created: {len(added_columns)}")


def encode_shipping_mode(
    train_data: pd.DataFrame, encoding_artifacts: dict[str, Any]
) -> None:
    """Step 4B: Apply explicit ordinal mapping for shipping service level."""
    print("\nGROUP B - MANUAL ORDINAL ENCODING")

    # KIYMETLİ VERİ BİLİMCİ NOTU: Sıralı (ordinal) kategorik kolonlar için rastgele kodlama yerine
    # manuel ağırlıklandırma kullanılır. Gönderim önceliği doğrudan iş mantığını yansıtır:
    # 'Same Day' en hızlı/öncelikli kargo iken 'Standard Class' en yavaş kargo türüdür.
    # Deterministik (belirlenmiş) bir harita kullanarak gelecekteki tahminleri de tutarlı hale getiriyoruz.
    shipping_mode_map = {
        "Same Day": 0,
        "First Class": 1,
        "Second Class": 2,
        "Standard Class": 3,
    }
    encoding_artifacts["shipping_mode_map"] = shipping_mode_map

    if "Shipping Mode" not in train_data.columns:
        print("  skipped missing column: Shipping Mode")
        return

    train_data["Shipping Mode"] = train_data["Shipping Mode"].map(shipping_mode_map)
    print("  applied Shipping Mode mapping:")
    print(f"  {shipping_mode_map}")


def target_encode_group_c(
    train_data: pd.DataFrame, encoding_artifacts: dict[str, Any]
) -> None:
    """Step 4C: Target encode high-cardinality nominal variables."""
    print("\nGROUP C - TARGET ENCODING USING late_delivery MEAN")

    if "late_delivery" not in train_data.columns:
        raise KeyError("late_delivery target is required for target encoding.")

    # KIYMETLİ VERİ BİLİMCİ NOTU: Çok yüksek kardinaliteye sahip nominal kategoriler ('Order Region', 'Customer State' vb.)
    # için One-Hot Encoding çok fazla kolon üretecektir. Bu durumu çözmek için hedef değişkenin ('late_delivery')
    # o kategori grubundaki ortalama oranı (mean target encoding) kullanılır. Örneğin 'Eastern Asia' bölgesinde geçmişte
    # geç teslimat oranı %65 ise, bu bölge adı 0.65 sayısal değeriyle temsil edilir. Bu sayede model karmaşıklığı 
    # ve overfitting azaltılırken bilgi kaybı yaşanmaz.
    target_encoding_columns = [
        "Category Name",
        "Order Region",
        "Customer State",
        "Order State",
        "Department Name",
    ]

    target_encoding_maps: dict[str, dict[str, float]] = {}

    for column in target_encoding_columns:
        if column not in train_data.columns:
            print(f"  skipped missing/already encoded column: {column}")
            continue

        encoding_map = train_data.groupby(column)["late_delivery"].mean()
        encoded_column = f"{column}_encoded"

        train_data[encoded_column] = train_data[column].map(encoding_map)
        train_data.drop(columns=column, inplace=True)

        map_dict = {
            str(key): round(float(value), 6)
            for key, value in encoding_map.sort_index().items()
        }
        target_encoding_maps[column] = map_dict

        print(f"\n  encoding map for {column}:")
        print(json.dumps(map_dict, ensure_ascii=True, indent=2))
        print(f"  created: {encoded_column}; dropped original: {column}")

    encoding_artifacts["target_encoding_maps"] = target_encoding_maps


def confirm_numeric_pass_through(train_data: pd.DataFrame) -> None:
    """Step 4D: Confirm prebuilt numeric calendar features remain untouched."""
    print("\nGROUP D - NUMERIC CALENDAR FEATURES PASSED THROUGH AS-IS")

    # KIYMETLİ VERİ BİLİMCİ NOTU: EDA aşamasında zaman damgalarından türetilmiş olan sayısal (int) 
    # özelliklerin (örneğin yıl, ay, gün, saat ve çeyrek) kodlama adımlarında bozulmadığını kontrol ediyoruz.
    # Bu değişkenler halihazırda sayısal oldukları için direkt olarak model eğitimine aktarılırlar.
    numeric_calendar_columns = [
        "order_year",
        "order_month",
        "order_week_day",
        "order_hour",
        "order_quarter",
    ]

    for column in numeric_calendar_columns:
        if column in train_data.columns:
            print(f"  passed through: {column} ({train_data[column].dtype})")
        else:
            print(f"  missing: {column}")


def encode_features(
    train_data: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, LabelEncoder]]:
    """Step 4: Apply feature encoding with reusable artifacts."""
    print_section("STEP 4 - LABEL ENCODING")

    encoding_artifacts: dict[str, Any] = {
        "encoding_strategy": "LabelEncoder per column",
        "target_model_type": "tree-based (RF, XGBoost, LightGBM)",
        "notes": (
            "LabelEncoder is appropriate for tree-based models. "
            "For linear models use OneHotEncoder instead. "
            "Load label_encoders.pkl to transform new data."
        ),
    }

    # Pipeline seviyesinde, ağaç tabanlı model stratejisine uygun olarak sütunları kodluyoruz.
    label_encoders = encode_with_label_encoders(train_data, encoding_artifacts)
    confirm_numeric_pass_through(train_data)

    return encoding_artifacts, label_encoders


def validate_final_dataset(train_data: pd.DataFrame) -> None:
    """Step 5: Print final modeling-table validation checks."""
    print_section("STEP 5 - FINAL VALIDATION")

    # KIYMETLİ VERİ BİLİMCİ NOTU: Modellemeye başlamadan önce verinin kalitesini kontrol etmek 
    # hayati önem taşır. Scikit-learn kütüphaneleri eksik veri (NaN) veya metin tabanlı (object) 
    # kolonlar aldığında hata fırlatır. Bu yüzden tüm verinin sayısal olduğundan ve eksik değer 
    # barındırmadığından emin oluyoruz.
    print(f"train_data shape: {train_data.shape}")

    print("\nDtype counts:")
    print(train_data.dtypes.value_counts())

    missing_values = train_data.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    print("\nMissing values:")
    if missing_values.empty:
        print("No missing values found.")
    else:
        print(missing_values)

    object_cols = train_data.select_dtypes(include=["object"]).columns.tolist()
    if object_cols:
        raise TypeError(f"Encoding incomplete. Object columns remain: {object_cols}")
    else:
        print("All columns are numeric. Encoding complete.")

    print_class_balance(train_data, "fraud")
    print_class_balance(train_data, "late_delivery")


def save_prepared_dataset(
    train_data: pd.DataFrame,
    encoding_artifacts: dict[str, Any],
    label_encoders: dict[str, LabelEncoder],
) -> None:
    """Step 6: Persist the prepared modeling data and reusable encodings."""
    print_section("STEP 6 - SAVE PREPARED DATASET")

    train_data.to_csv(OUTPUT_DATA_PATH, index=False)

    with OUTPUT_ENCODING_PATH.open("w", encoding="utf-8") as file:
        json.dump(encoding_artifacts, file, indent=2, ensure_ascii=False)

    joblib.dump(label_encoders, "label_encoders.pkl")

    print(
        f"Pipeline complete. Shape: {train_data.shape}. "
        f"Saved as {OUTPUT_DATA_PATH}"
    )
    print(f"Encoding artifacts saved as {OUTPUT_ENCODING_PATH}")
    print("Label encoders saved as label_encoders.pkl")


def prepare_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full CRISP-DM data-preparation pipeline.

    Returns
    -------
    pd.DataFrame
        Prepared `train_data`, ready for train/test split and model training.
    """
    train_data = create_training_copy(df)
    create_binary_targets(train_data)
    drop_modeling_exclusions(train_data)
    encoding_artifacts, label_encoders = encode_features(train_data)
    validate_final_dataset(train_data)
    save_prepared_dataset(train_data, encoding_artifacts, label_encoders)
    return train_data


# Notebook usage:
# train_data = prepare_training_data(df)
