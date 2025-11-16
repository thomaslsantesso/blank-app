from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import plotly.express as px
import logging
from datetime import datetime
from email.message import EmailMessage
import mimetypes


# ----------------------------
# Config
# ----------------------------
ROOT = Path(".")
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
REPORTS_DIR = OUTPUT_DIR / "reports"

QUOTES_PATH = DATA_DIR / "Quotes.csv"
ACCOUNTS_PATH = DATA_DIR / "Accounts.csv"

for d in (OUTPUT_DIR, PLOTS_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(OUTPUT_DIR / "data_loader.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ----------------------------
# Helpers - column/normalization
# ----------------------------
def _find_col(df: pd.DataFrame, candidates):
    """Returns the real column name of the first match among candidates (case-insensitive)."""
    lc = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lc:
            return lc[cand.lower()]
    return None

def normalize_numeric_series(s: pd.Series, remove_pct: bool = False) -> pd.Series:
    """Cleans strings and converts to float, accepting comma and %."""
    out = s.astype(str).fillna("").str.strip()
    if remove_pct:
        out = out.str.replace("%", "", regex=False)
    out = out.str.replace(r"[^\d\-,\.]", "", regex=True).str.replace(",", ".", regex=False)
    return pd.to_numeric(out, errors="coerce")

# ----------------------------
# Loaders
# ----------------------------
def load_quotes(path: Path) -> pd.DataFrame:
    logging.info(f"Reading Quotes: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Quotes file not found: {path}")

    df = pd.read_csv(path, dtype=str, low_memory=False).rename(columns=str.strip)

    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()

    date_col = _find_col(df, ["QuoteDate", "Date", "quote_date", "Quote Date"])
    net_col = _find_col(df, ["NetPrice", "Net_Price", "netprice", "Net Price"])
    msrp_col = _find_col(df, ["MSRP"])
    qty_col = _find_col(df, ["Quantity", "Qty"])
    disc_col = _find_col(df, ["DiscountPercent", "Discount%", "Discount %"])
    prod_col = _find_col(df, ["ProductName", "Product"])
    partner_col = _find_col(df, ["PartnerAccountName", "PartnerAccount", "AccountName", "CustomerName"])

    if date_col:
        df["QuoteDate"] = pd.to_datetime(df[date_col], errors="coerce")
    if net_col:
        df["NetPrice"] = normalize_numeric_series(df[net_col])
    if msrp_col:
        df["MSRP"] = normalize_numeric_series(df[msrp_col])
    if qty_col:
        df["Quantity"] = pd.to_numeric(df[qty_col].astype(str).str.replace(",", "."), errors="coerce").fillna(0).astype(int)
    if disc_col:
        df["DiscountPercent"] = normalize_numeric_series(df[disc_col], remove_pct=True).fillna(0.0)
    if prod_col:
        df["ProductName"] = df[prod_col].astype(str)
    if partner_col:
        df["PartnerAccountName"] = df[partner_col].astype(str)

    if "NetPrice" not in df.columns and "MSRP" in df.columns and "Quantity" in df.columns:
        df["NetPrice"] = (df["MSRP"].astype(float) * df["Quantity"].astype(int))

    if "QuoteDate" in df.columns:
        df["Year"] = df["QuoteDate"].dt.year
        df["Month"] = df["QuoteDate"].dt.month
        df["MonthStart"] = df["QuoteDate"].dt.to_period("M").dt.to_timestamp()
        df["Quarter"] = df["QuoteDate"].dt.to_period("Q").astype(str)

    logging.info(f"Quotes loaded: {len(df)} rows")
    return df


def load_accounts(path: Path) -> pd.DataFrame:
    logging.info(f"Reading Accounts: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Accounts file not found: {path}")

    df = pd.read_csv(path, dtype=str, low_memory=False).rename(columns=str.strip)

    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()

    acc_name = _find_col(df, ["AccountName", "Account"])
    acc_id = _find_col(df, ["AccountId", "Id"])
    acc_type = _find_col(df, ["Type", "AccountType"])
    city = _find_col(df, ["City"])
    state = _find_col(df, ["State", "Region"])
    zipc = _find_col(df, ["ZipCode", "PostalCode"])
    street = _find_col(df, ["Street", "Address"])
    contact = _find_col(df, ["PrimaryContact", "Contact"])
    contact_email = _find_col(df, ["ContactEmail", "Email"])
    parent = _find_col(df, ["ParentAccount", "Parent"])

    if acc_name:
        df["AccountName"] = df[acc_name]
    if acc_id:
        df["AccountId"] = df[acc_id]
    if acc_type:
        df["AccountType"] = df[acc_type]
    if city:
        df["City"] = df[city]
    if state:
        df["State"] = df[state]
    if zipc:
        df["ZipCode"] = df[zipc]
    if street:
        df["Street"] = df[street]
    if contact:
        df["PrimaryContact"] = df[contact]
    if contact_email:
        df["ContactEmail"] = df[contact_email]
    if parent:
        df["ParentAccount"] = df[parent]

    logging.info(f"Accounts loaded: {len(df)} rows")
    return df
# ----------------------------
# Email
# ----------------------------
def create_eml_with_pdf(
    to_address: str,
    subject: str,
    body: str,
    pdf_path: Path,
    output_eml_path: Path,
    from_address: str = "no-reply@company.com",
):
    """
    Generates a .eml file PRE-FILLED with:
    - recipient
    - subject
    - body text
    - PDF attachment

    User only needs to double-click the .eml and press "Send".
    """
    msg = EmailMessage()
    msg["From"] = from_address
    msg["To"] = to_address
    msg["Subject"] = subject
    msg.set_content(body)

    pdf_path = Path(pdf_path)
    with open(pdf_path, "rb") as f:
        pdf_data = f.read()

    mime_type, _ = mimetypes.guess_type(str(pdf_path))
    if mime_type is None:
        mime_type = "application/pdf"
    main_type, sub_type = mime_type.split("/")

    msg.add_attachment(pdf_data, maintype=main_type, subtype=sub_type, filename=pdf_path.name)

    # write .eml file
    with open(output_eml_path, "wb") as f:
        f.write(bytes(msg))

    print(f".eml saved at: {output_eml_path}")

def get_client_email(accounts_df: pd.DataFrame, account_name: str) -> Optional[str]:
    df = accounts_df
    if "AccountName" not in df.columns or "ContactEmail" not in df.columns:
        return None


    row = df[df["AccountName"].astype(str).str.lower() == str(account_name).lower()]
    if row.empty:
        return None
    return row.iloc[0]["ContactEmail"]


##############################################
# 4) ADD function: generate_pdf_and_eml_for_client
##############################################
def generate_pdf_and_eml_for_client(
    df_merged: pd.DataFrame,
    df_accounts: pd.DataFrame,
    account_name: str,
    team_email: Optional[str] = None
):
    """
    1. Filters merged data by client
    2. Generates PDF
    3. Creates a .eml file ready to send to client and optionally team
    """

    sub = df_merged[df_merged["PartnerAccountName"].astype(str).str.lower() == account_name.lower()]
    if sub.empty:
        print(f"No quotes found for: {account_name}")
        return None

    stamp_base = REPORTS_DIR / f"client_{account_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Generate PDF
    pdf_path = generate_pdf_report(sub, stamp_base)

    # Find e-mail
    client_email = get_client_email(df_accounts, account_name)

    subject = f"Commercial Report - {account_name}"
    body = f"Attached is the commercial PDF report for client: {account_name}."

    # create .eml for client
    if client_email:
        eml_client = stamp_base.with_suffix("_client.eml")
        create_eml_with_pdf(
            to_address=client_email,
            subject=subject,
            body=body,
            pdf_path=pdf_path,
            output_eml_path=eml_client,
        )

    # create .eml for internal team
    if team_email:
        eml_team = stamp_base.with_suffix("_team.eml")
        create_eml_with_pdf(
            to_address=team_email,
            subject=f"[TEAM] {subject}",
            body=f"Internal copy — Client: {account_name}",
            pdf_path=pdf_path,
            output_eml_path=eml_team,
        )

    return pdf_path
# ----------------------------
# Merge & enrich
# ----------------------------
def merge_quotes_accounts(df_quotes: pd.DataFrame, df_accounts: pd.DataFrame) -> pd.DataFrame:
    left = df_quotes.copy()
    right = df_accounts.copy()

    left["__partner_norm"] = left.get("PartnerAccountName", "").astype(str).str.strip().str.lower()
    right["__acc_norm"] = right.get("AccountName", "").astype(str).str.strip().str.lower()

    merged = left.merge(
        right,
        how="left",
        left_on="__partner_norm",
        right_on="__acc_norm",
        suffixes=("", "_acc")
    )

    null_ratio = merged["AccountName"].isna().mean() if "AccountName" in merged.columns else 1.0
    logging.info(f"Merge null ratio (AccountName): {null_ratio:.3f}")

    merged.drop(columns=["__partner_norm", "__acc_norm"], errors="ignore", inplace=True)
    return merged

# ----------------------------
# Aggregations & KPIs
# ----------------------------
def compute_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    kpis = {}
    kpis["total_revenue"] = float(df["NetPrice"].sum()) if "NetPrice" in df.columns else 0.0
    kpis["total_quantity"] = int(df["Quantity"].sum()) if "Quantity" in df.columns else 0
    kpis["unique_clients"] = int(df["PartnerAccountName"].nunique()) if "PartnerAccountName" in df.columns else 0
    kpis["unique_products"] = int(df["ProductName"].nunique()) if "ProductName" in df.columns else 0
    kpis["avg_ticket"] = (kpis["total_revenue"] / kpis["total_quantity"]) if kpis["total_quantity"] else 0.0
    kpis["avg_discount"] = float(df["DiscountPercent"].mean()) if "DiscountPercent" in df.columns else 0.0
    kpis["states_covered"] = int(df["State"].nunique()) if "State" in df.columns else 0
    kpis["cities_covered"] = int(df["City"].nunique()) if "City" in df.columns else 0
    logging.info(f"KPIs computed: revenue={kpis['total_revenue']}, clients={kpis['unique_clients']}")
    return kpis

def top_n_by(df: pd.DataFrame, by: str, value_col: str = "NetPrice", n: int = 10) -> pd.DataFrame:
    return df.groupby(by, as_index=False)[value_col].sum().sort_values(value_col, ascending=False).head(n)

def revenue_by_period(df: pd.DataFrame, period: str = "M") -> pd.DataFrame:
    if "QuoteDate" not in df.columns:
        return pd.DataFrame()

    if period == "D":
        agg = df.groupby(df["QuoteDate"].dt.date, as_index=False)["NetPrice"].sum().rename(columns={"QuoteDate": "Period", "NetPrice": "Revenue"})
        agg["Period"] = pd.to_datetime(agg["Period"])
        return agg

    if period == "M":
        agg = df.groupby(df["MonthStart"], as_index=False)["NetPrice"].sum().rename(columns={"MonthStart": "Period", "NetPrice": "Revenue"})
        return agg

    if period == "Q":
        agg = df.groupby("Quarter", as_index=False)["NetPrice"].sum().rename(columns={"Quarter": "Period", "NetPrice": "Revenue"})
        return agg

    if period == "Y":
        agg = df.groupby("Year", as_index=False)["NetPrice"].sum().rename(columns={"Year": "Period", "NetPrice": "Revenue"})
        return agg

    return pd.DataFrame()

# ----------------------------
# Plots (saved)
# ----------------------------
def plot_revenue_monthly(df: pd.DataFrame, out_path: Path):
    agg = revenue_by_period(df, "M")
    if agg.empty:
        return None

    fig = px.line(agg, x="Period", y="Revenue", title="Monthly Revenue", markers=True)
    out_html = out_path.with_suffix(".monthly.html")
    fig.write_html(str(out_html))
    logging.info(f"Saved interactive monthly HTML: {out_html}")

    plt.figure(figsize=(8,4))
    plt.plot(agg["Period"], agg["Revenue"], marker="o")
    plt.title("Monthly Revenue")
    plt.xlabel("Month")
    plt.ylabel("Revenue")
    plt.grid(True)
    png_path = out_path.with_suffix(".monthly.png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()

    return png_path


def plot_top_clients(df: pd.DataFrame, out_path: Path, n:int=10):
    top = top_n_by(df, "PartnerAccountName", "NetPrice", n)
    if top.empty:
        return None

    fig = px.bar(top.sort_values("NetPrice"), x="NetPrice", y="PartnerAccountName",
                 orientation="h", title=f"Top {n} Clients by Revenue")
    out_html = out_path.with_suffix(".clients.html")
    fig.write_html(str(out_html))
    logging.info(f"Saved interactive top clients: {out_html}")

    plt.figure(figsize=(8,4))
    plt.barh(top["PartnerAccountName"], top["NetPrice"])
    plt.title(f"Top {n} Clients")
    plt.xlabel("Revenue")
    plt.tight_layout()
    png_path = out_path.with_suffix(".clients.png")
    plt.savefig(png_path, dpi=150)
    plt.close()

    return png_path


def plot_discount_distribution(df: pd.DataFrame, out_path: Path):
    if "DiscountPercent" not in df.columns:
        return None

    fig = px.histogram(df, x="DiscountPercent", nbins=40, title="Discount Distribution (%)")
    out_html = out_path.with_suffix(".discount.html")
    fig.write_html(str(out_html))
    logging.info(f"Saved interactive discount histogram: {out_html}")

    plt.figure(figsize=(6,4))
    plt.hist(df["DiscountPercent"].dropna(), bins=40)
    plt.title("Discount Distribution (%)")
    plt.xlabel("Discount (%)")
    plt.tight_layout()
    png_path = out_path.with_suffix(".discount.png")
    plt.savefig(png_path, dpi=150)
    plt.close()

    return png_path


def plot_state_choropleth(df: pd.DataFrame, out_path: Path):
    if "State" not in df.columns or "NetPrice" not in df.columns:
        return None

    state_sum = df.groupby("State", as_index=False)["NetPrice"].sum()

    fig = px.choropleth(
        state_sum,
        locations="State",
        locationmode="USA-states",
        color="NetPrice",
        color_continuous_scale="Blues",
        title="Revenue by State (USA)"
    )
    out_html = out_path.with_suffix(".states_map.html")
    fig.write_html(str(out_html))
    logging.info(f"Saved interactive state choropleth: {out_html}")

    return out_html


def plot_city_scatter_geo(df: pd.DataFrame, out_path: Path):
    if not {"City","State","NetPrice"}.issubset(df.columns):
        return None

    city_sum = df.groupby(["City","State"], as_index=False)["NetPrice"].sum()

    fig = px.scatter_geo(
        city_sum,
        locations="State",
        locationmode="USA-states",
        size="NetPrice",
        hover_name="City",
        hover_data=["NetPrice"],
        scope="usa",
        title="Revenue by City (bubble size)"
    )
    out_html = out_path.with_suffix(".city_scatter.html")
    fig.write_html(str(out_html))
    logging.info(f"Saved interactive city scatter: {out_html}")

    return out_html

def plot_revenue_by_city_bar(df: pd.DataFrame, out_path: Path):
    """
    Plota receita por cidade agrupada por estado (sem bolhas), 
    usando barras horizontais, para cada estado.
    """
    if not {"City", "State", "NetPrice"}.issubset(df.columns):
        return None

    # Agrega receita por cidade
    city_sum = df.groupby(["State", "City"], as_index=False)["NetPrice"].sum()

    # Ordena dentro de cada estado
    city_sum = city_sum.sort_values(["State", "NetPrice"], ascending=[True, False])

    # Cria gráfico de barras horizontais por estado
    fig = px.bar(
        city_sum,
        x="NetPrice",
        y="City",
        color="State",
        orientation="h",
        title="Revenue by City (Grouped by State)",
        hover_data={"NetPrice":":,.2f", "State":True}
    )

    fig.update_layout(
        yaxis={"categoryorder":"total ascending"},
        barmode="stack",
        height=600 + len(city_sum) * 10  # ajusta altura para muitas cidades
    )

    out_html = out_path.with_suffix(".city_by_state.html")
    fig.write_html(str(out_html))
    logging.info(f"Saved Revenue by City bar chart: {out_html}")
    return out_html
# ----------------------------
# PDF report
# ----------------------------
def generate_pdf_report(df: pd.DataFrame, out_path: Path, meta: Optional[Dict[str,Any]]=None):
    pngs = []

    p1 = plot_revenue_monthly(df, out_path.with_suffix(".monthly"))
    if p1: pngs.append(p1)

    p2 = plot_top_clients(df, out_path.with_suffix(".clients"))
    if p2: pngs.append(p2)

    p3 = plot_discount_distribution(df, out_path.with_suffix(".discount"))
    if p3: pngs.append(p3)

    pdf_path = out_path.with_suffix(".pdf")
    with PdfPages(pdf_path) as pdf:

        fig = plt.figure(figsize=(8.27, 11.69))
        plt.axis("off")

        txt = "COMMERCIAL REPORT\n\n"
        if meta is None:
            meta = {}

        kpis = compute_kpis(df)
        txt += f"Unique clients: {kpis['unique_clients']}\n"
        txt += f"Unique products: {kpis['unique_products']}\n"
        txt += f"Total revenue (NetPrice): $ {kpis['total_revenue']:,.2f}\n"
        txt += f"Total quantity: {kpis['total_quantity']}\n"
        txt += f"Average ticket: $ {kpis['avg_ticket']:,.2f}\n"
        txt += f"Average discount: {kpis['avg_discount']:.2f}%\n"
        txt += f"States served: {kpis['states_covered']}\n"

        plt.text(0.01, 0.99, txt, va="top", fontsize=12, family="monospace")
        pdf.savefig()
        plt.close()

        for p in pngs:
            try:
                img = plt.imread(p)
                fig = plt.figure(figsize=(8.27, 11.69))
                plt.imshow(img)
                plt.axis("off")
                pdf.savefig()
                plt.close()
            except Exception as e:
                logging.error(f"Error adding image to PDF: {e}")

    logging.info(f"PDF report generated: {pdf_path}")
    return pdf_path

# ============================================================
# NEW: LIGHT GEOCODING + GEO TABLES + GEO PLOTS (ADDED ONLY)
# ============================================================

US_STATE_CENTROIDS = {
    "AL": (32.806671, -86.791130),
    "AK": (61.370716, -152.404419),
    "AZ": (33.729759, -111.431221),
    "AR": (34.969704, -92.373123),
    "CA": (36.116203, -119.681564),
    "CO": (39.059811, -105.311104),
    "CT": (41.597782, -72.755371),
    "DE": (39.318523, -75.507141),
    "FL": (27.766279, -81.686783),
    "GA": (33.040619, -83.643074),
    "HI": (21.094318, -157.498337),
    "ID": (44.240459, -114.478828),
    "IL": (40.349457, -88.986137),
    "IN": (39.849426, -86.258278),
    "IA": (42.011539, -93.210526),
    "KS": (38.526600, -96.726486),
    "KY": (37.668140, -84.670067),
    "LA": (31.169546, -91.867805),
    "ME": (44.693947, -69.381927),
    "MD": (39.063946, -76.802101),
    "MA": (42.230171, -71.530106),
    "MI": (43.326618, -84.536095),
    "MN": (45.694454, -93.900192),
    "MS": (32.741646, -89.678696),
    "MO": (38.456085, -92.288368),
    "MT": (46.921925, -110.454353),
    "NE": (41.125370, -98.268082),
    "NV": (38.313515, -117.055374),
    "NH": (43.452492, -71.563896),
    "NJ": (40.298904, -74.521011),
    "NM": (34.840515, -106.248482),
    "NY": (42.165726, -74.948051),
    "NC": (35.630066, -79.806419),
    "ND": (47.528912, -99.784012),
    "OH": (40.388783, -82.764915),
    "OK": (35.565342, -96.928917),
    "OR": (44.572021, -122.070938),
    "PA": (40.590752, -77.209755),
    "RI": (41.680893, -71.511780),
    "SC": (33.856892, -80.945007),
    "SD": (44.299782, -99.438828),
    "TN": (35.747845, -86.692345),
    "TX": (31.054487, -97.563461),
    "UT": (40.150032, -111.862434),
    "VT": (44.045876, -72.710686),
    "VA": (37.769337, -78.169968),
    "WA": (47.400902, -121.490494),
    "WV": (38.491226, -80.954453),
    "WI": (44.268543, -89.616508),
    "WY": (42.755966, -107.302490)
}

def add_geo_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds latitude & longitude using:
    1. State centroid (always works)
    2. Optional city-level jitter (to visually separate cities)
    """
    df = df.copy()

    # STATE-LEVEL
    df["lat"] = df["State"].map(lambda s: US_STATE_CENTROIDS.get(str(s).upper(), (None, None))[0])
    df["lon"] = df["State"].map(lambda s: US_STATE_CENTROIDS.get(str(s).upper(), (None, None))[1])

    # CITY JITTER → separation within the state
    df["lat"] = df["lat"] + (df["City"].astype(str).str.__hash__() % 100) * 0.0005
    df["lon"] = df["lon"] + (df["City"].astype(str).str.__hash__() % 100) * 0.0005

    return df

def plot_geo_city_map(df: pd.DataFrame, out_path: Path):
    """Bubble map by city, using approximate geolocation."""
    if not {"City", "State", "NetPrice", "lat", "lon"}.issubset(df.columns):
        return None

    g = df.groupby(["City","State","lat","lon"], as_index=False)["NetPrice"].sum()

    fig = px.scatter_geo(
        g,
        lat="lat",
        lon="lon",
        size="NetPrice",
        hover_name="City",
        hover_data=["State", "NetPrice"],
        scope="usa",
        title="Revenue by City (Geo Map)"
    )

    out_html = out_path.with_suffix(".city_geo.html")
    fig.write_html(str(out_html))
    logging.info(f"Saved GEO city map: {out_html}")
    return out_html

def plot_geo_address_detail(df: pd.DataFrame, out_path: Path):
    """Plot points using street/zip as unique identifiers (approximate only)."""
    if not {"Street","City","State","lat","lon"}.issubset(df.columns):
        return None

    df2 = df.dropna(subset=["Street"]).copy()

    if df2.empty:
        return None

    df2["size"] = df2["NetPrice"].fillna(0.0)

    fig = px.scatter_geo(
        df2,
        lat="lat",
        lon="lon",
        hover_name="Street",
        hover_data=["City","State","NetPrice"],
        size="size",
        scope="usa",
        title="Revenue by Address (approx. GEO)"
    )

    out_html = out_path.with_suffix(".address_geo.html")
    fig.write_html(str(out_html))
    logging.info(f"Saved GEO address map: {out_html}")
    return out_html

# Hook to inject GEO data right after merge
def load_all_geo(quotes_path: Path = QUOTES_PATH, accounts_path: Path = ACCOUNTS_PATH):
    dq, da, dm = load_all(quotes_path, accounts_path)
    dm = add_geo_coordinates(dm)
    return dq, da, dm


# ----------------------------
# Exports CSV / Excel
# ----------------------------
def export_dataframes(df: pd.DataFrame, prefix: str = "report"):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = REPORTS_DIR / f"{prefix}_{stamp}.csv"
    xlsx_path = REPORTS_DIR / f"{prefix}_{stamp}.xlsx"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_excel(xlsx_path, index=False, engine="openpyxl")
    logging.info(f"Exported CSV: {csv_path} and XLSX: {xlsx_path}")
    return csv_path, xlsx_path

# ----------------------------
# Top-level orchestration
# ----------------------------
def load_all(quotes_path: Path = QUOTES_PATH, accounts_path: Path = ACCOUNTS_PATH) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dq = load_quotes(quotes_path)
    da = load_accounts(accounts_path)
    dm = merge_quotes_accounts(dq, da)

    if "NetPrice" in dm.columns:
        dm["NetPrice"] = pd.to_numeric(dm["NetPrice"], errors="coerce").fillna(0.0)

    logging.info("load_all finished")
    return dq, da, dm

def generate_all_reports(df_merged: pd.DataFrame, prefix: str = "summary"):
    stamp_base = REPORTS_DIR / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out = {}

    out["csv"], out["xlsx"] = export_dataframes(df_merged, prefix)

    out["monthly_png"] = plot_revenue_monthly(df_merged, stamp_base)
    out["top_clients_png"] = plot_top_clients(df_merged, stamp_base)
    out["discount_png"] = plot_discount_distribution(df_merged, stamp_base)
    out["states_map_html"] = plot_state_choropleth(df_merged, stamp_base)
    out["city_by_state_html"] = plot_revenue_by_city_bar(df_merged, stamp_base)

    out["pdf"] = generate_pdf_report(df_merged, stamp_base)

    logging.info(f"generate_all_reports completed: {out}")
    return out

# ----------------------------
# CLI / quick test
# ----------------------------
if __name__ == "__main__":
    print("Loading Quotes and Accounts, generating analyses...")
    try:
        dq, da, dm = load_all()
    except Exception as e:
        logging.exception("Failed to load data")
        raise

    print(f"Loaded Quotes: {len(dq)} rows, Accounts: {len(da)} rows, Merged: {len(dm)} rows")
    kpis = compute_kpis(dm)
    print("KPIs:", kpis)

    print("Generating reports (CSV, XLSX, HTML maps, PDF)...")
    results = generate_all_reports(dm, prefix="summary_report")
    print("Done. Outputs:")
    for k, v in results.items():
        print(f" - {k}: {v}")
    print("Logs in:", OUTPUT_DIR / "data_loader.log")
