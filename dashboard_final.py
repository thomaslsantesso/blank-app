import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import logging
import io
from email.message import EmailMessage
import mimetypes
import matplotlib
matplotlib.use("Agg")  # backend headless para gerar imagens/pdfs no servidor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Config paths (ajuste se quiser)
ROOT = Path('.')
DATA_DIR = ROOT / 'data'
OUTPUT_DIR = ROOT / 'outputs'
REPORTS_DIR = OUTPUT_DIR / 'reports'
for d in (OUTPUT_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

QUOTES_PATH = DATA_DIR / 'Quotes.csv'
ACCOUNTS_PATH = DATA_DIR / 'Accounts.csv'

# Logging
logging.basicConfig(filename=str(OUTPUT_DIR / 'dashboard.log'), level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------
# Helpers (loader & normalizer)
# ---------------------------
def _find_col(df: pd.DataFrame, candidates):
    lc = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lc:
            return lc[cand.lower()]
    return None

def normalize_numeric_series(s: pd.Series, remove_pct: bool = False) -> pd.Series:
    out = s.astype(str).fillna('').str.strip()
    if remove_pct:
        out = out.str.replace('%', '', regex=False)
    out = out.str.replace(r"[^\d\-,\.]", "", regex=True).str.replace(",", ".", regex=False)
    return pd.to_numeric(out, errors='coerce')

@st.cache_data(ttl=3600)
def load_quotes(path: Path) -> pd.DataFrame:
    logging.info(f'Reading Quotes from {path}')
    if not path.exists():
        raise FileNotFoundError(f'Quotes not found: {path}')
    df = pd.read_csv(path, dtype=str, low_memory=False).rename(columns=str.strip)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()

    date_col = _find_col(df, ['QuoteDate','Date','quote_date'])
    net_col = _find_col(df, ['NetPrice','Net_Price','Net Price','netprice'])
    msrp_col = _find_col(df, ['MSRP','msrp'])
    qty_col = _find_col(df, ['Quantity','Qty','quantity'])
    disc_col = _find_col(df, ['DiscountPercent','Discount_Percent','Discount %','Discount%','discountpercent'])
    prod_col = _find_col(df, ['ProductName','Product','product_name'])
    partner_col = _find_col(df, ['PartnerAccountName','PartnerAccount','AccountName','CustomerName','Customer'])

    if date_col:
        df['QuoteDate'] = pd.to_datetime(df[date_col], errors='coerce')
    if net_col:
        df['NetPrice'] = normalize_numeric_series(df[net_col]).fillna(0.0)
    if msrp_col:
        df['MSRP'] = normalize_numeric_series(df[msrp_col]).fillna(0.0)
    if qty_col:
        df['Quantity'] = pd.to_numeric(df[qty_col].astype(str).str.replace(',', '.'), errors='coerce').fillna(0).astype(int)
    if disc_col:
        df['DiscountPercent'] = normalize_numeric_series(df[disc_col], remove_pct=True).fillna(0.0)
    if prod_col:
        df['ProductName'] = df[prod_col].astype(str)
    if partner_col:
        df['PartnerAccountName'] = df[partner_col].astype(str)

    # fallback compute NetPrice if missing
    if 'NetPrice' not in df.columns and 'MSRP' in df.columns and 'Quantity' in df.columns:
        df['NetPrice'] = (df['MSRP'] * df['Quantity']).fillna(0.0)

    # period cols
    if 'QuoteDate' in df.columns:
        df['Year'] = df['QuoteDate'].dt.year
        df['Month'] = df['QuoteDate'].dt.month
        df['MonthStart'] = df['QuoteDate'].dt.to_period('M').dt.to_timestamp()
        df['Quarter'] = df['QuoteDate'].dt.to_period('Q').astype(str)
        df['Bimester'] = ((df['Month'] - 1) // 2 + 1).astype('Int64')
        df['Semester'] = ((df['Month'] - 1) // 6 + 1).astype('Int64')

    logging.info(f'Quotes loaded: {len(df)} rows')
    return df

@st.cache_data(ttl=3600)
def load_accounts(path: Path) -> pd.DataFrame:
    logging.info(f'Reading Accounts from {path}')
    if not path.exists():
        raise FileNotFoundError(f'Accounts not found: {path}')
    df = pd.read_csv(path, dtype=str, low_memory=False).rename(columns=str.strip)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()

    acc_name = _find_col(df, ['AccountName','Account','Account Name','accountname'])
    acc_id = _find_col(df, ['AccountId','Id','accountid'])
    acc_type = _find_col(df, ['Type','AccountType','RecordType','type'])
    city = _find_col(df, ['City','city'])
    state = _find_col(df, ['State','Region','Estado','UF'])
    zipc = _find_col(df, ['ZipCode','PostalCode','Zip','zipcode'])
    street = _find_col(df, ['Street','Address','Address1'])
    contact = _find_col(df, ['PrimaryContact','Contact','Primary Contact'])
    contact_email = _find_col(df, ['ContactEmail','Email','contactemail'])
    parent = _find_col(df, ['ParentAccount','Parent','Parent Account'])

    if acc_name: df['AccountName'] = df[acc_name].astype(str)
    if acc_id: df['AccountId'] = df[acc_id].astype(str)
    if acc_type: df['AccountType'] = df[acc_type].astype(str)
    if city: df['City'] = df[city].astype(str)
    if state: df['State'] = df[state].astype(str)
    if zipc: df['ZipCode'] = df[zipc].astype(str)
    if street: df['Street'] = df[street].astype(str)
    if contact: df['PrimaryContact'] = df[contact].astype(str)
    if contact_email: df['ContactEmail'] = df[contact_email].astype(str)
    if parent: df['ParentAccount'] = df[parent].astype(str)

    logging.info(f'Accounts loaded: {len(df)} rows')
    return df

def merge_quotes_accounts(df_quotes: pd.DataFrame, df_accounts: pd.DataFrame) -> pd.DataFrame:
    left = df_quotes.copy()
    right = df_accounts.copy()
    left['__partner_norm'] = left.get('PartnerAccountName', '').astype(str).str.strip().str.lower()
    right['__acc_norm'] = right.get('AccountName', '').astype(str).str.strip().str.lower()
    merged = left.merge(right, how='left', left_on='__partner_norm', right_on='__acc_norm', suffixes=('','_acc'))
    merged.drop(columns=[c for c in ['__partner_norm','__acc_norm'] if c in merged.columns], inplace=True)
    logging.info(f'Merged rows: {len(merged)}')
    return merged
def create_eml_with_pdf(
    to_address: str,
    subject: str,
    body: str,
    pdf_path: Path,
    output_eml_path: Path,
    from_address: str = "no-reply@company.com",
):
    """
    Cria um .eml pré-preenchido com PDF anexado.
    Usuário pode abrir no Outlook/Thunderbird/Windows Mail e só apertar Enviar.
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

    # grava .eml
    with open(output_eml_path, "wb") as f:
        f.write(bytes(msg))

    print(f".eml saved at: {output_eml_path}")


def get_client_email(accounts_df: pd.DataFrame, account_name: str) -> Optional[str]:
    """
    Retorna ContactEmail dado AccountName. Se accounts_df for vazio retorna None.
    """
    if accounts_df is None or accounts_df.empty:
        return None
    if "AccountName" not in accounts_df.columns or "ContactEmail" not in accounts_df.columns:
        return None
    row = accounts_df[accounts_df["AccountName"].astype(str).str.lower() == str(account_name).lower()]
    if row.empty:
        return None
    return row.iloc[0]["ContactEmail"]


def generate_pdf_report_for_client(df_client: pd.DataFrame, account_name: str, out_path_base: Path) -> Path:
    """
    Gera um PDF simples com:
     - página 1: KPIs (texto)
     - página 2: gráfico mensal (se possível)
     - página 3: top produtos (se possível)
    Retorna caminho do PDF gerado.
    """
    out_path_base = Path(out_path_base)
    out_path_base.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = out_path_base.with_suffix(".pdf")

    # prepara KPIs
    k = compute_kpis(df_client)

    # abre PDF e escreve páginas
    with PdfPages(pdf_path) as pdf:
        # Página 1: texto KPIs
        fig = plt.figure(figsize=(8.27, 11.69))
        plt.axis("off")
        txt = f"Commercial Report — {account_name}\n\n"
        txt += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        txt += f"Unique clients (in slice): {k.get('unique_clients')}\n"
        txt += f"Unique products: {k.get('unique_products')}\n"
        txt += f"Total revenue (NetPrice): $ {k.get('total_revenue'):,.2f}\n"
        txt += f"Total quantity: {k.get('total_quantity') if 'total_quantity' in k else k.get('total_qty')}\n"
        txt += f"Average ticket: $ {k.get('avg_ticket'):,.2f}\n"
        txt += f"Average discount: {k.get('avg_discount'):,.2f}%\n"
        plt.text(0.01, 0.99, txt, va="top", fontsize=12, family="monospace")
        pdf.savefig()
        plt.close()

        # Página 2: receita mensal (se dados existirem)
        if "MonthStart" in df_client.columns and "NetPrice" in df_client.columns:
            monthly = df_client.groupby("MonthStart", as_index=False)["NetPrice"].sum().sort_values("MonthStart")
            fig, ax = plt.subplots(figsize=(8.27, 4))
            ax.plot(monthly["MonthStart"], monthly["NetPrice"], marker="o")
            ax.set_title("Monthly Revenue")
            ax.set_xlabel("Month")
            ax.set_ylabel("Revenue")
            ax.grid(True)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # Página 3: top produtos
        if "ProductName" in df_client.columns and "NetPrice" in df_client.columns:
            prod = df_client.groupby("ProductName", as_index=False)["NetPrice"].sum().sort_values("NetPrice", ascending=False).head(10)
            if not prod.empty:
                fig, ax = plt.subplots(figsize=(8.27, 4))
                ax.barh(prod["ProductName"][::-1], prod["NetPrice"][::-1])
                ax.set_title("Top Products (revenue)")
                ax.set_xlabel("Revenue")
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

    print(f"PDF generated: {pdf_path}")
    return pdf_path


def generate_pdf_and_eml_for_client(
    df_merged: pd.DataFrame,
    df_accounts: pd.DataFrame,
    account_name: str,
    team_email: Optional[str] = None,
):
    """
    Gera PDF do cliente filtrado e cria .eml para cliente (se e-mail existir) e opcionalmente para o time.
    """
    # filtra por PartnerAccountName (ou AccountName quando Partner faltar)
    key = "PartnerAccountName" if "PartnerAccountName" in df_merged.columns else "AccountName"
    sub = df_merged[df_merged[key].astype(str).str.lower() == account_name.lower()]
    if sub.empty:
        print(f"No quotes found for: {account_name}")
        return None

    safe_name = account_name.replace(" ", "_").replace("/", "_")
    stamp_base = REPORTS_DIR / f"client_{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # gera pdf
    pdf_path = generate_pdf_report_for_client(sub, account_name, stamp_base)

    # pega email do account
    client_email = get_client_email(df_accounts, account_name)

    subject = f"Commercial Report - {account_name}"
    body = f"Attached is the commercial PDF report for client: {account_name}."

    # inicializa variáveis (sempre definidas)
    eml_client = None
    eml_team = None

    # cria .eml para cliente
    if client_email:
        eml_client = Path(str(stamp_base) + "_client.eml")
        create_eml_with_pdf(
            to_address=client_email,
            subject=subject,
            body=body,
            pdf_path=pdf_path,
            output_eml_path=eml_client,
        )

    # cria .eml para o time
    if team_email:
        eml_team = Path(str(stamp_base) + "_team.eml")
        create_eml_with_pdf(
            to_address=team_email,
            subject=f"[TEAM] {subject}",
            body=f"Internal copy — Client: {account_name}",
            pdf_path=pdf_path,
            output_eml_path=eml_team,
        )

    return {"pdf": pdf_path, "eml_client": eml_client, "eml_team": eml_team}

# ---------------------------
# FUNÇÃO MAPA DOS EUA COM PIZZAS POR ESTADO
# ---------------------------
import plotly.express as px
import plotly.graph_objects as go

def choropleth_states_pie(df: pd.DataFrame, value_col: str = 'NetPrice'):
    """Gera um choropleth dos EUA por estado e permite gerar gráfico de pizza por estado"""
    if not {'State','City',value_col}.issubset(df.columns):
        return None, None

    # Receita por estado
    states_agg = df.groupby('State', as_index=False)[value_col].sum()

    # Choropleth EUA completo
    fig_map = go.Figure(go.Choropleth(
        locations=states_agg['State'],
        z=states_agg[value_col],
        locationmode='USA-states',
        colorscale='Blues',
        marker_line_color='white',  # fronteiras dos estados
        marker_line_width=1,
        hovertemplate='<b>%{location}</b><br>Total: $%{z:,.0f}<extra></extra>'
    ))

    fig_map.update_layout(
        title_text='Receita por Estado (EUA)',
        geo_scope='usa',
        geo=dict(
            showlakes=True, lakecolor='rgb(255,255,255)',
        )
    )

    # Função interna para gerar pizza de um estado específico
    def pie_by_state(state: str):
        df_state = df[df['State'] == state]
        if df_state.empty:
            return None
        city_agg = df_state.groupby('City', as_index=False)[value_col].sum()
        fig_pie = px.pie(city_agg, names='City', values=value_col,
                         title=f'Receita por Cidade em {state}',
                         hole=0.3)
        fig_pie.update_traces(textinfo='percent+label')
        return fig_pie

    return fig_map, pie_by_state

def compute_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    k = {}
    k['total_revenue'] = float(df['NetPrice'].sum()) if 'NetPrice' in df.columns else 0.0
    k['total_qty'] = int(df['Quantity'].sum()) if 'Quantity' in df.columns else 0
    k['unique_clients'] = int(df['PartnerAccountName'].nunique()) if 'PartnerAccountName' in df.columns else 0
    k['unique_products'] = int(df['ProductName'].nunique()) if 'ProductName' in df.columns else 0
    k['avg_ticket'] = (k['total_revenue'] / k['total_qty']) if k['total_qty'] else 0.0
    k['avg_discount'] = float(df['DiscountPercent'].mean()) if 'DiscountPercent' in df.columns else 0.0
    k['states'] = int(df['State'].nunique()) if 'State' in df.columns else 0
    k['cities'] = int(df['City'].nunique()) if 'City' in df.columns else 0
    return k

@st.cache_data(ttl=3600)
def load_all(quotes_path: Path = QUOTES_PATH, accounts_path: Path = ACCOUNTS_PATH) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dq = load_quotes(quotes_path)
    da = load_accounts(accounts_path) if accounts_path and accounts_path.exists() else pd.DataFrame()
    dm = merge_quotes_accounts(dq, da) if not da.empty else dq.copy()
    if 'NetPrice' in dm.columns:
        dm['NetPrice'] = pd.to_numeric(dm['NetPrice'], errors='coerce').fillna(0.0)
    return dq, da, dm

# ---------------------------
# Streamlit app
# ---------------------------
st.set_page_config(page_title='Dashboard Motorola — Quotes', layout='wide', initial_sidebar_state='expanded')
st.title('📊 Dashboard Executivo — Quotes & Vendas (Quotes + Accounts)')

# Caminho absoluto onde estão os CSVs
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
QUOTES_PATH = DATA_DIR / "Quotes.csv"
ACCOUNTS_PATH = DATA_DIR / "Accounts.csv"

# Verifica existência dos arquivos
if not QUOTES_PATH.exists():
    st.error(f'Quotes.csv não encontrado em {QUOTES_PATH}')
    st.stop()

if not ACCOUNTS_PATH.exists():
    st.warning(f'Accounts.csv não encontrado em {ACCOUNTS_PATH}, dados de contas serão ignorados')
    ACCOUNTS_PATH = None

# Carregar os dados automaticamente
try:
    df_quotes, df_accounts, df = load_all(QUOTES_PATH, ACCOUNTS_PATH)
except Exception as e:
    st.error(f'Erro ao carregar dados: {e}')
    st.stop()

# Carregar os dados automaticamente
try:
    df_quotes, df_accounts, df = load_all(QUOTES_PATH, ACCOUNTS_PATH)
except Exception as e:
    st.error(f'Erro ao carregar dados: {e}')
    st.stop()

# Basic detection
if 'QuoteDate' not in df.columns:
    st.warning('Coluna QuoteDate ausente — gráficos temporais podem não aparecer.')

client_col = 'PartnerAccountName' if 'PartnerAccountName' in df.columns else ('QuoteName' if 'QuoteName' in df.columns else None)
prod_col = 'ProductName' if 'ProductName' in df.columns else None
region_col = 'State' if 'State' in df.columns else None

# Global filters
with st.expander('Filtros globais (mostrar/ocultar)', expanded=True):
    cols = st.columns([2,2,2,2])
    if 'QuoteDate' in df.columns:
        min_date = df['QuoteDate'].min().date()
        max_date = df['QuoteDate'].max().date()
        start_date, end_date = cols[0].date_input('Período', value=(min_date, max_date))
    else:
        start_date, end_date = None, None

    if client_col:
        quotes_clients = pd.Series(df[client_col].dropna().unique()) if client_col in df.columns else pd.Series(dtype=str)
        accounts_clients = pd.Series(df_accounts['AccountName'].dropna().unique()) if (df_accounts is not None and not df_accounts.empty and 'AccountName' in df_accounts.columns) else pd.Series(dtype=str)
        clients = sorted(pd.unique(pd.concat([quotes_clients, accounts_clients], ignore_index=True)))
    else:
        clients = []
    client_sel = cols[1].multiselect('Cliente', options=clients, default=None)

    prods = sorted(df[prod_col].dropna().unique()) if prod_col else []
    prod_sel = cols[2].multiselect('Produto', options=prods, default=None)

    regs = sorted(df[region_col].dropna().unique()) if region_col else []
    region_sel = cols[3].multiselect('Região / State', options=regs, default=None)

# Sidebar advanced filters
st.sidebar.markdown('---')
st.sidebar.subheader('Filtros avançados')
if 'DiscountPercent' not in df.columns:
    df['DiscountPercent'] = 0.0
if 'Quantity' not in df.columns:
    df['Quantity'] = 0

try:
    disc_min, disc_max = st.sidebar.slider('Desconto (%)', 0.0, 100.0, (float(df['DiscountPercent'].min()), float(df['DiscountPercent'].max())))
except Exception:
    disc_min, disc_max = 0.0, 100.0

try:
    qmin = int(df['Quantity'].min())
    qmax = int(df['Quantity'].max())
    qty_min, qty_max = st.sidebar.slider('Quantidade', qmin, qmax, (qmin, qmax))
except Exception:
    qty_min, qty_max = None, None

# Apply filters safely
working = df.copy()
if start_date and end_date and 'QuoteDate' in working.columns:
    working = working[(working['QuoteDate'] >= pd.to_datetime(start_date)) & (working['QuoteDate'] <= pd.to_datetime(end_date))]
if client_sel:
    working = working[working[client_col].isin(client_sel)]
if prod_sel:
    working = working[working[prod_col].isin(prod_sel)]
if region_sel and region_col:
    working = working[working[region_col].isin(region_sel)]

if 'DiscountPercent' in working.columns:
    working['DiscountPercent'] = pd.to_numeric(working['DiscountPercent'], errors='coerce').fillna(0.0)
    working = working[(working['DiscountPercent'] >= float(disc_min)) & (working['DiscountPercent'] <= float(disc_max))]
if qty_min is not None and 'Quantity' in working.columns:
    working['Quantity'] = pd.to_numeric(working['Quantity'], errors='coerce').fillna(0).astype(int)
    working = working[(working['Quantity'] >= int(qty_min)) & (working['Quantity'] <= int(qty_max))]

if 'NetPrice' in working.columns:
    working['NetPrice'] = pd.to_numeric(working['NetPrice'], errors='coerce').fillna(0.0)

# KPIs
k = compute_kpis(working)
kcols = st.columns(4)
kcols[0].metric('💰 Receita total (USD)', f'${k["total_revenue"]:,.2f}')
kcols[1].metric('📦 Quantidade total', f'{k["total_qty"]:,}')
kcols[2].metric('🎯 Ticket médio (USD)', f'${k["avg_ticket"]:,.2f}')
kcols[3].metric('% Desconto médio', f'{k["avg_discount"]:.2f}%')

st.markdown('---')

# Tabs
tabs = st.tabs(['Visão Geral','Clientes','Produtos','Descontos & Preço','Análise Temporal','Mapa (EUA)','Tabela Detalhada'])

with tabs[0]:
    st.header('Visão Geral')
    left, right = st.columns([3,1])
    with left:
        if 'MonthStart' in working.columns and 'NetPrice' in working.columns:
            monthly = working.groupby('MonthStart', as_index=False)['NetPrice'].sum()
            fig = px.line(monthly, x='MonthStart', y='NetPrice', title='Receita Mensal', markers=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('Dados temporais insuficientes para gráfico mensal.')

        if client_col and 'NetPrice' in working.columns:
            clients_sum = working.groupby(client_col, as_index=False)['NetPrice'].sum().sort_values('NetPrice', ascending=False).head(10)
            figc = px.bar(clients_sum, x='NetPrice', y=client_col, orientation='h', title='Top 10 Clientes')
            st.plotly_chart(figc, use_container_width=True)
        else:
            st.info('Coluna cliente ausente.')
    with right:
        st.subheader('Resumo Rápido')
        st.metric('Clientes únicos', f'{k["unique_clients"]}')
        st.metric('Produtos únicos', f'{k["unique_products"]}')
        st.metric('Estados atendidos', f'{k["states"]}')

with tabs[1]:
    st.header('Clientes')
    if client_col:
        agg = working.groupby(client_col, as_index=False).agg(Receita=('NetPrice','sum'), Quantidade=('Quantity','sum'))
        st.dataframe(agg.sort_values('Receita', ascending=False).head(200))
        st.download_button('Exportar clientes (CSV)', agg.to_csv(index=False).encode('utf-8-sig'), file_name='clientes_resumo.csv')
    else:
        st.info('Coluna cliente não encontrada.')

with tabs[2]:
    st.header('Produtos')
    if prod_col:
        prod_df = working.groupby(prod_col, as_index=False).agg(Receita=('NetPrice','sum'), Quantidade=('Quantity','sum'))
        st.dataframe(prod_df.sort_values('Receita', ascending=False).head(200))
        figp = px.bar(prod_df.sort_values('Receita', ascending=False).head(20), x='Receita', y=prod_col, orientation='h', title='Top Produtos por Receita')
        st.plotly_chart(figp, use_container_width=True)
        st.download_button('Exportar produtos (CSV)', prod_df.to_csv(index=False).encode('utf-8-sig'), file_name='produtos_resumo.csv')
    else:
        st.info('Coluna ProductName ausente.')

with tabs[3]:
    st.header("📉 Descontos & Preço")

    st.markdown("""
    Esta seção mostra **como os descontos estão sendo aplicados**, como isso afeta o **preço final** e 
    quais **parceiros ou produtos** puxam mais o desconto médio.  
    Use esta aba para identificar rapidamente:
    - Produtos com “desconto agressivo demais”
    - Oportunidades de **recuperar margem**
    - Parceiros que **negociam mais desconto** do que a média
    """)

    # =============== CÁLCULOS BASE ===============
    df_disc = df.copy()
    if 'DiscountPercent' in df_disc.columns:
        df_disc['Discount_Percent'] = df_disc['DiscountPercent']
    elif 'Discount' in df_disc.columns and 'List Price' in df_disc.columns:
        df_disc['Discount_Percent'] = df_disc['Discount'] / df_disc['List Price']
    elif 'ListPrice' in df_disc.columns and 'NetPrice' in df_disc.columns:
        df_disc['Discount_Percent'] = (df_disc['ListPrice'] - df_disc['NetPrice']) / df_disc['ListPrice']
    else:
        st.error("Não encontrei nenhuma coluna que permita calcular Discount_Percent.")
        st.stop()

    avg_discount = df_disc['Discount_Percent'].mean()
    avg_net_price = df_disc['NetPrice'].mean()

    top_discount_products = (
        df_disc.groupby('ProductName')['Discount_Percent'].mean()
        .sort_values(ascending=False)
        .head(5)
    )

    top_discount_partners = (
        df_disc.groupby('PartnerAccountName')['Discount_Percent'].mean()
        .sort_values(ascending=False)
        .head(5)
    )

    # =============== MÉTRICAS EM CARDS ===============
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Desconto Médio",
            f"{avg_discount:.2f}%",
            help="Percentual médio de desconto aplicado sobre o preço de lista."
        )

    with col2:
        st.metric(
            "Preço Médio Líquido",
            f"${avg_net_price:,.2f}",
            help="Preço final médio após aplicação dos descontos."
        )

    with col3:
        st.metric(
            "Produto + Sensível a Desconto",
            f"{top_discount_products.index[0]}",
            help="Produto que recebe o maior desconto médio."
        )

    st.markdown("---")

    # =============== CARDS EXPLICATIVOS ===============
    with st.expander("ℹ️ Como interpretar estes dados"):
        st.markdown("""
        **Desconto Médio** → mostra o “apetite de desconto” da região.  
        Valores muito altos podem indicar **erosão de margem**.

        **Preço Médio Líquido** → importante para entender o quanto estamos nos afastando do preço de lista.

        **Top Produtos / Parceiros com Maior Desconto** → ajuda a identificar  
        - onde existe espaço para **recuperar preço**  
        - parceiros que negociam mais que os demais  
        - produtos onde o argumento de valor está fraco
        """)

    # =============== GRÁFICO 1 – Produtos com maior desconto ===============
    st.subheader("📦 Produtos com Maior Desconto Médio")

    st.bar_chart(
        top_discount_products,
        height=300
    )

    # =============== GRÁFICO 2 – Parceiros com maior desconto ===============
    st.subheader("🤝 Parceiros que Mais Puxam Desconto")

    st.bar_chart(
        top_discount_partners,
        height=300
    )

    st.markdown("---")

    # =============== TABELA RESUMO FINAL ===============
    st.subheader("📊 Tabela Resumo — Top 15 Maiores Descontos")
    df_table = (
        df_disc[['ProductName', 'PartnerAccountName','DiscountPercent', 'NetPrice']].copy()
        .sort_values('DiscountPercent', ascending=False)
        .head(15)
    )
    st.dataframe(df_table, use_container_width=True)

with tabs[4]:
    st.header('Análise Temporal')
    if 'QuoteDate' in working.columns and 'NetPrice' in working.columns:
        period = st.selectbox('Agregação', ['Dia','Mês','Bimestre','Trimestre','Semestre'])
        if period == 'Dia':
            working['Period'] = working['QuoteDate'].dt.date  # cria a coluna 'Period' explicitamente
            agg = working.groupby('Period', as_index=False)['NetPrice'].sum()
            agg['Period'] = pd.to_datetime(agg['Period'])  # converte para datetime
            x = 'Period'
        elif period == 'Mês':
            agg = working.groupby('MonthStart', as_index=False)['NetPrice'].sum()
            x = 'MonthStart'
        elif period == 'Bimestre':
            agg = working.groupby(['Year','Bimester'], as_index=False)['NetPrice'].sum()
            agg['Period'] = agg['Year'].astype(str) + '-B' + agg['Bimester'].astype(str)
            x = 'Period'
        elif period == 'Trimestre':
            agg = working.groupby('Quarter', as_index=False)['NetPrice'].sum()
            x = 'Quarter'
        else:
            agg = working.groupby(['Year','Semester'], as_index=False)['NetPrice'].sum()
            agg['Period'] = agg['Year'].astype(str) + '-S' + agg['Semester'].astype(str)
            x = 'Period'
        st.plotly_chart(px.line(agg, x=x, y='NetPrice', markers=True, title='Receita ao longo do tempo'), use_container_width=True)
    else:
        st.info('Dados temporais ausentes.')

with tabs[5]:
    st.header('Mapa — Estados e Cidades (EUA)')
    st.markdown('Comparação geográfica completa de vendas por Estado.')

    # Chama a função
    fig_map, pie_fn = choropleth_states_pie(working)
    if fig_map:
        st.plotly_chart(fig_map, use_container_width=True)

    # Seleciona o estado para ver pizza
    selected_state = st.selectbox('Escolha o estado para ver pizza das cidades', sorted(working['State'].dropna().unique()))
    if selected_state:
        fig_pie = pie_fn(selected_state)
        if fig_pie:
            st.plotly_chart(fig_pie, use_container_width=True)

with tabs[6]:
    st.header('Tabela Detalhada')
    st.write('Filtre e exporte os dados brutos abaixo:')
    st.dataframe(working.head(1000))
    csv_data = working.to_csv(index=False).encode('utf-8-sig')
    st.download_button('Exportar CSV completo', data=csv_data, file_name=f'quotes_export_{datetime.now().strftime("%Y%m%d_%H%M")}.csv')

st.markdown('---')
st.subheader('Exportar / Gerar Relatório')
col1, col2, col3 = st.columns(3)
with col1:
    st.download_button('Exportar CSV (filtro atual)', data=working.to_csv(index=False).encode('utf-8-sig'), file_name=f'report_{datetime.now().strftime("%Y%m%d_%H%M")}.csv')
with col2:
    bio = io.BytesIO()
    working.to_excel(bio, index=False, engine='openpyxl')
    st.download_button('Exportar XLSX (filtro atual)', data=bio.getvalue(), file_name=f'report_{datetime.now().strftime("%Y%m%d_%H%M")}.xlsx')
with col3:
    if st.button('Gerar CSV resumão (Reports folder)'):
        csv_path = REPORTS_DIR / f'filtered_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        working.to_csv(csv_path, index=False, encoding='utf-8-sig')
        st.success(f'CSV salvo em: {csv_path}')
# --- ADICIONAR UI: Gerar PDF + .eml (cole após o bloco de export já existente) ---
st.markdown('---')
st.subheader('Gerar Relatório PDF (.pdf) e preparar E-mail (.eml) — Option A (manual send)')

# client list seguro (recria caso scope anterior não exista)
clients_list = sorted(df[client_col].dropna().unique()) if client_col and client_col in df.columns else []

col_a, col_b = st.columns([2,2])
with col_a:
    selected_client = st.selectbox('Cliente para gerar relatório', options=[''] + clients_list, index=0)
    team_email_input = st.text_input('E-mail da equipe (opcional) — receberá cópia .eml', value='')

with col_b:
    if st.button('Gerar PDF + .eml para cliente selecionado'):
        if not selected_client:
            st.error('Selecione um cliente antes.')
        else:
            # cria nome seguro para keys
            safe_name = "".join([c if c.isalnum() else "_" for c in selected_client])[:64]

            try:
                res = generate_pdf_and_eml_for_client(
                    df, df_accounts, selected_client, team_email=team_email_input or None
                )
            except Exception as e:
                logging.exception("Erro ao gerar pdf/eml")
                st.error(f"Erro ao gerar relatório: {e}")
                st.exception(e)
                res = None

            # Resultado
            if not res:
                st.warning('Nenhum dado encontrado para o cliente selecionado ou ocorreu um erro.')
            else:
                pdf_path = Path(res.get("pdf")) if res.get("pdf") else None
                st.success(f"PDF gerado: {pdf_path.name if pdf_path else '—'}")

                # Baixar PDF
                if pdf_path and pdf_path.exists():
                    with open(pdf_path, "rb") as f:
                        pdf_bytes = f.read()
                    st.download_button(
                        'Baixar PDF',
                        data=pdf_bytes,
                        file_name=pdf_path.name,
                        mime='application/pdf',
                        key=f"download_pdf_{safe_name}"
                    )
                else:
                    st.warning('PDF não encontrado no disco.')

                # Baixar .eml (cliente)
                eml_client_path = Path(res.get("eml_client")) if res.get("eml_client") else None
                if eml_client_path and eml_client_path.exists():
                    with open(eml_client_path, "rb") as f:
                        eml_bytes = f.read()
                    st.download_button(
                        'Baixar .eml (cliente)',
                        data=eml_bytes,
                        file_name=eml_client_path.name,
                        mime='message/rfc822',
                        key=f"download_eml_client_{safe_name}"
                    )

                # Baixar .eml (team)
                eml_team_path = Path(res.get("eml_team")) if res.get("eml_team") else None
                if eml_team_path and eml_team_path.exists():
                    with open(eml_team_path, "rb") as f:
                        eml_team_bytes = f.read()
                    st.download_button(
                        'Baixar .eml (team)',
                        data=eml_team_bytes,
                        file_name=eml_team_path.name,
                        mime='message/rfc822',
                        key=f"download_eml_team_{safe_name}"
                    )

# gerenciar geração em massa para clientes filtrados
st.caption('Ou gere para todos os clientes atualmente visíveis no filtro abaixo:')
if st.button('Gerar PDFs + .eml para TODOS os clientes visíveis'):
    saved = []
    missing = []
    # obtém lista de clientes únicos na tabela filtrada `working`
    target_clients = sorted(working[client_col].dropna().unique()) if client_col and client_col in working.columns else []
    if not target_clients:
        st.warning('Nenhum cliente visível nos filtros.')
    else:
        progress = st.progress(0)
        n = len(target_clients)
        for i, c in enumerate(target_clients, start=1):
            r = generate_pdf_and_eml_for_client(df, df_accounts, c, team_email=team_email_input or None)
            if r:
                saved.append(r)
            else:
                missing.append(c)
            progress.progress(int(i / n * 100))
        st.success(f'Gerados {len(saved)} relatórios. {len(missing)} sem dados.')
        # mostra os primeiros 5 PDFs para download
        for item in saved[:5]:
            if item and 'pdf' in item:
                with open(item['pdf'], 'rb') as f:
                    st.download_button(f"Baixar {Path(item['pdf']).name}", data=f.read(), file_name=Path(item['pdf']).name, mime='application/pdf')
# --- fim UI ---


st.caption('Dashboard integrado: Quotes + Accounts — mapas e segmentações por City/State/Type')
