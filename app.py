# app.py - FINAL VERSION (REVISED & STABLE)

import base64
import io
import pandas as pd
import numpy as np
import warnings

# Plotly and Dash components
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, State, dash_table, ctx
import dash_bootstrap_components as dbc

# Machine Learning components for Derived Importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Helper Functions & Initial Setup ---

PROCESSED_DATA_CACHE = {}

try:
    name_mapping_df = pd.read_csv('CompanyName_Aspect_Attribute.csv', header=0)
except FileNotFoundError:
    print("Error: 'CompanyName_Aspect_Attribute.csv' not found. Please ensure the file is in the same directory.")
    name_mapping_df = pd.DataFrame(columns=['Type', 'Category', 'Original_Name', 'Short_Name'])

def process_data(contents, filename):
    if contents is None:
        return None
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_excel(io.BytesIO(decoded), engine='openpyxl')
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None
    score_cols = [col for col in df.columns if col.startswith(('T_H', 'H10', 'T_A', 'T_L1', 'T_D2', 'D3_'))]
    for col in score_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if 'T_L1' in col:
            df[col] = df[col].where((df[col] >= 0) & (df[col] <= 10), np.nan)
        else:
            df[col] = df[col].where((df[col] >= 1) & (df[col] <= 10), np.nan)
    d2_cols_rename = {col: col.replace('T_D2_', 'T_H9_') for col in df.columns if col.startswith('T_D2_')}
    df_renamed = df.rename(columns=d2_cols_rename)
    h1_h8_cols = [col for col in df_renamed.columns if col.startswith('T_H') and not col.startswith('T_H9_')]
    df_h1_h8_long = df_renamed.melt(
        id_vars=['SbjNum', 'Segmen', 'Region', 'Provinsi'], value_vars=h1_h8_cols,
        var_name='Question_Code', value_name='Satisfaction_Score'
    )
    extracted_h1_h8 = df_h1_h8_long['Question_Code'].str.extract(r'T_H(\d+)_(\d+)_(\d+)')
    df_h1_h8_long[['Aspect_Num', 'Company_Num', 'Attr_Num']] = extracted_h1_h8
    df_h1_h8_long.dropna(subset=['Aspect_Num', 'Company_Num', 'Attr_Num'], inplace=True)
    df_h1_h8_long['Company_Num'] = pd.to_numeric(df_h1_h8_long['Company_Num'])
    s16_names = df[['SbjNum', 'S16_1', 'S16_2', 'S16_3']].melt(
        id_vars='SbjNum', var_name='Company_Col', value_name='Original_Name'
    )
    s16_names['Company_Num'] = s16_names['Company_Col'].str.extract(r'S16_(\d+)').astype(int)
    df_company = pd.merge(df_h1_h8_long, s16_names[['SbjNum', 'Company_Num', 'Original_Name']], on=['SbjNum', 'Company_Num'])
    h9_cols = [col for col in df_renamed.columns if col.startswith('T_H9_')]
    df_h9_satisfaction = df_renamed.melt(
        id_vars=['SbjNum', 'Segmen', 'Region', 'Provinsi'], value_vars=h9_cols,
        var_name='Question_Code', value_name='Satisfaction_Score'
    )
    extracted_h9_pos = df_h9_satisfaction['Question_Code'].str.extract(r'T_H9_(\d+)_(\d+)')
    df_h9_satisfaction[['App_Eval_Position', 'Attr_Num']] = extracted_h9_pos
    df_h9_satisfaction.dropna(subset=['App_Eval_Position', 'Attr_Num'], inplace=True)
    df_h9_satisfaction['App_Eval_Position'] = pd.to_numeric(df_h9_satisfaction['App_Eval_Position'])
    df_h9_satisfaction['Aspect_Num'] = 9
    d1c_cols = [f'D1C_{i}' for i in range(1, 4) if f'D1C_{i}' in df.columns]
    df_app_final = pd.DataFrame()
    if d1c_cols:
        df_d1c_choices = df.melt(id_vars=['SbjNum'], value_vars=d1c_cols, var_name='App_Eval_Position_Str', value_name='Original_Name')
        df_d1c_choices['App_Eval_Position'] = df_d1c_choices['App_Eval_Position_Str'].str.extract(r'D1C_(\d+)').astype(int)
        df_d1c_choices.dropna(subset=['Original_Name'], inplace=True)
        df_app_merged = pd.merge(df_h9_satisfaction, df_d1c_choices[['SbjNum', 'App_Eval_Position', 'Original_Name']], on=['SbjNum', 'App_Eval_Position'], how='inner')
        df_app = df_app_merged.copy()
    else:
        df_app = pd.DataFrame()
    def enrich_dataframe(df_to_enrich, category_name):
        if df_to_enrich.empty: return pd.DataFrame()
        a_cols = [col for col in df.columns if col.startswith('T_A')]
        df_a_long = df.melt(id_vars=['SbjNum'], value_vars=a_cols, var_name='Question_Code', value_name='Importance_Score')
        extracted_a = df_a_long['Question_Code'].str.extract(r'T_A(\d+)_(\d+)')
        df_a_long[['Aspect_Num', 'Attr_Num']] = extracted_a
        df_a_long.dropna(subset=['Aspect_Num', 'Attr_Num'], inplace=True)
        for col in ['Aspect_Num', 'Attr_Num']:
            df_to_enrich[col] = pd.to_numeric(df_to_enrich[col])
            df_a_long[col] = pd.to_numeric(df_a_long[col])
        enriched_df = pd.merge(df_to_enrich, df_a_long, on=['SbjNum', 'Aspect_Num', 'Attr_Num'], how='left')
        mapping_subset = name_mapping_df[name_mapping_df['Category'] == category_name]
        company_short_names = mapping_subset[mapping_subset['Type'] == 'Company'][['Original_Name', 'Short_Name']].rename(columns={'Short_Name':'Company_Short_Name'})
        enriched_df = pd.merge(enriched_df, company_short_names, on='Original_Name', how='left')
        enriched_df['Attribute_Code'] = enriched_df.apply(lambda row: f"A{int(row['Aspect_Num'])}.{int(row['Attr_Num'])}" if pd.notna(row['Aspect_Num']) and pd.notna(row['Attr_Num']) else None, axis=1)
        enriched_df['Aspect_Code'] = enriched_df.apply(lambda row: f"H{int(row['Aspect_Num'])}" if pd.notna(row['Aspect_Num']) else None, axis=1)
        aspect_short_names = name_mapping_df[name_mapping_df['Type'] == 'Aspect'][['Original_Name', 'Short_Name']].rename(columns={'Original_Name':'Aspect_Code', 'Short_Name':'Aspect_Short_Name'})
        enriched_df = pd.merge(enriched_df, aspect_short_names, on='Aspect_Code', how='left')
        attribute_short_names = name_mapping_df[name_mapping_df['Type'] == 'Attribute'][['Original_Name', 'Short_Name']].rename(columns={'Original_Name':'Attribute_Code', 'Short_Name':'Attribute_Short_Name'})
        enriched_df = pd.merge(enriched_df, attribute_short_names, on='Attribute_Code', how='left')
        enriched_df['Aspect_Short_Name'] = enriched_df['Aspect_Short_Name'].fillna(enriched_df['Aspect_Code'])
        enriched_df['Attribute_Short_Name'] = enriched_df['Attribute_Short_Name'].fillna(enriched_df['Attribute_Code'])
        return enriched_df
    df_company_final = enrich_dataframe(df_company, 'Pesticide_Company')
    df_app_final = enrich_dataframe(df_app, 'Digital_App')
    if not df_company_final.empty:
        h10_nps_long = s16_names.copy()
        h10_nps_scores_melted = df.melt(id_vars='SbjNum', value_vars=['H10_1', 'H10_2', 'H10_3', 'T_L1_1', 'T_L1_2', 'T_L1_3'], var_name='Score_Col', value_name='Score')
        h10_nps_scores_melted['Type'] = np.where(h10_nps_scores_melted['Score_Col'].str.startswith('H10'), 'Overall_CSAT', 'NPS_Score')
        h10_nps_scores_melted['Company_Num'] = h10_nps_scores_melted['Score_Col'].str.extract(r'_(\d+)').astype(int)
        h10_nps_scores_pivoted = h10_nps_scores_melted.pivot_table(index=['SbjNum', 'Company_Num'], columns='Type', values='Score').reset_index()
        h10_nps_long = pd.merge(h10_nps_long, h10_nps_scores_pivoted, on=['SbjNum', 'Company_Num'])
        df_company_final = pd.merge(df_company_final, h10_nps_long[['SbjNum', 'Original_Name', 'Overall_CSAT', 'NPS_Score']], on=['SbjNum', 'Original_Name'], how='left')
        df_company_final['NPS_Category'] = pd.cut(df_company_final['NPS_Score'], bins=[-1, 6, 8, 10], labels=['Detractors', 'Passives', 'Promoters'])
    if not df_app_final.empty:
        d3_cols_present = [col for col in df.columns if col.startswith('D3_')]
        if d3_cols_present:
            d3_scores = df.melt(id_vars='SbjNum', value_vars=d3_cols_present, var_name='Score_Col', value_name='Overall_CSAT_App')
            d3_scores['App_Eval_Position'] = d3_scores['Score_Col'].str.extract(r'_(\d+)').astype(int)
            df_app_final = pd.merge(df_app_final, d3_scores[['SbjNum', 'App_Eval_Position', 'Overall_CSAT_App']], on=['SbjNum', 'App_Eval_Position'], how='left')
        else:
            df_app_final['Overall_CSAT_App'] = np.nan
    return {"company": df_company_final.to_json(orient='split'), "app": df_app_final.to_json(orient='split')}

def filter_dataframe(df, segments, regions, provinces):
    if df.empty: return df
    if not segments and not regions and not provinces: return df
    if segments: df = df[df['Segmen'].isin(segments)]
    if regions: df = df[df['Region'].isin(regions)]
    if provinces: df = df[df['Provinsi'].isin(provinces)]
    return df

def calculate_derived_importance(df, mode, aspect_name=None):
    if mode == 'company':
        dv = 'Overall_CSAT'
        pivot_level = 'Attribute_Short_Name'
        df_model = df[df['Aspect_Short_Name'] == aspect_name].copy() if aspect_name else df.copy()
    else:
        dv = 'Overall_CSAT_App'
        df_model = df.copy()
        pivot_level = 'Attribute_Short_Name'
    if df_model.empty: return pd.DataFrame()
    df_wide = df_model.pivot_table(index=['SbjNum', 'Original_Name'], columns=pivot_level, values='Satisfaction_Score', aggfunc='mean')
    overall_scores = df.drop_duplicates(subset=['SbjNum', 'Original_Name'])
    overall_scores = overall_scores.set_index(['SbjNum', 'Original_Name'])[[dv]]
    df_wide = df_wide.join(overall_scores, on=['SbjNum', 'Original_Name'])
    iv_cols = df_wide.columns.drop(dv, errors='ignore')
    df_wide.dropna(subset=[dv] + iv_cols.tolist(), inplace=True)
    if df_wide.empty or len(df_wide) < 10: return pd.DataFrame()
    X = df_wide[iv_cols]
    y = df_wide[dv]
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    importance_df = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
    scaler = MinMaxScaler(feature_range=(1, 10))
    importance_df['Derived_Importance'] = scaler.fit_transform(importance_df[['importance']])
    return importance_df[['feature', 'Derived_Importance']]

app = Dash(__name__, external_stylesheets=[dbc.themes.LUX], suppress_callback_exceptions=True)
server = app.server

navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                dbc.Row(
                    [
                        dbc.Col(html.Img(src="/assets/logo.png", height="45px")),
                        dbc.Col(dbc.NavbarBrand("Customer Satisfaction Dashboard", className="ms-2")),
                    ],
                    align="center",
                    className="g-0",
                ),
                href="#",
                style={"textDecoration": "none"},
            ),
        ]
    ),
    color="#002060",
    dark=True,
)

app.layout = html.Div([
    dcc.Store(id='processed-data-store'),
    dcc.Download(id="download-performance-data-xlsx"),
    dcc.Download(id="download-kpi-data"),
    dcc.Download(id="download-attribute-data"),
    dcc.Download(id="download-stated-derived-data"),
    dcc.Download(id="download-dataframe-xlsx"),
    navbar,
    dbc.Container([
        dbc.Row([
            dbc.Col(
                dcc.Upload(
                    id='upload-data',
                    children=html.Div(['Drag and Drop or ', html.A('Select an Excel File')]),
                    style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px 0'}
                ), width=6
            ),
            dbc.Col(
                 dbc.Card(dbc.CardBody(dcc.RadioItems(
                    id='mode-selector',
                    options=[
                        {'label': 'Company Performance Evaluation', 'value': 'company'},
                        {'label': 'Digital App Evaluation', 'value': 'app'}
                    ],
                    value='company',
                ))), width=6
            )
        ], className="mt-4"),
        html.Hr(),
        dbc.Row([
            # --- START: REVISED LAYOUT FOR SIDEBAR ---
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Global Filters", className="card-title"),
                        html.Hr(),
                        # This Div will be populated by the render_main_content callback
                        html.Div(id='dynamic-filters-placeholder'),
                        html.Br(),
                        # Buttons are now static and disabled by default
                        html.Div([
                            dbc.Button("Download Company Data", id="btn-download-company", color="success", className="me-1 mt-2", disabled=True),
                            dbc.Button("Download App Data", id="btn-download-app", color="info", className="mt-2", disabled=True),
                        ]),
                    ])
                ], style={'backgroundColor': '#f8f9fa'}),
                width=3
            ),
            # --- END: REVISED LAYOUT ---
            dbc.Col(id='tabs-container', width=9)
        ])
    ], fluid=True)
])

# --- START: REVISED CALLBACK ---
@app.callback(
    Output('dynamic-filters-placeholder', 'children'),
    Output('tabs-container', 'children'),
    Input('processed-data-store', 'data')
)
def render_main_content(json_data):
    if json_data is None:
        filter_content = html.P("Upload data to enable filters.", className="text-muted")
        tabs = dbc.Card(dbc.CardBody(dbc.Alert("Please upload an Excel file to begin analysis.", color="info")))
        return filter_content, tabs

    filter_content = html.Div([
        html.P("1. Filter by Segment:", className="card-text"),
        dcc.Dropdown(id='filter-segment', placeholder="All Segments", multi=True),
        html.Br(),
        html.P("2. Filter by Region:", className="card-text"),
        dcc.Dropdown(id='filter-region', placeholder="All Regions", multi=True),
        html.Br(),
        html.P("3. Filter by Province:", className="card-text"),
        dcc.Dropdown(id='filter-province', placeholder="All Provinces", multi=True),
    ])

    tabs = dcc.Tabs(id="tabs-main", children=[
        dcc.Tab(label="KPI Summary", children=[
            dbc.Card(dbc.CardBody([
                dbc.Row([
                    dbc.Col(html.H5("Select entities to display:", className="card-title"), width=True),
                    dbc.Col(dbc.Button("Download KPI Data", id="btn-download-kpi", color="secondary", size="sm"), width="auto"),
                ], align="center"),
                dcc.Dropdown(id='entity-select', multi=True, placeholder="Select entities to compare...", style={'fontSize': '14px'}),
                html.Hr(),
                dbc.Row(dbc.Col(dcc.Graph(id='csat-dist-chart'), width=12)),
                dbc.Row(dbc.Col(dcc.Graph(id='nps-dist-chart'), id='nps-chart-col', width=12))
            ]), className="mt-3")
        ]),
        dcc.Tab(label="CSAT by Attribute", children=[
            dbc.Card(dbc.CardBody([
                dbc.Row([
                    dbc.Col(html.H5("Select entities to display:", className="card-title"), width=True),
                    dbc.Col(dbc.Button("Download Attribute Data", id="btn-download-attribute", color="secondary", size="sm"), width="auto"),
                ], align="center"),
                dcc.Dropdown(id='entity-select-attribute', multi=True, placeholder="Select entities to compare...", style={'fontSize': '14px'}),
                html.Hr(),
                dcc.Loading(html.Div(id='snake-charts-container'))
            ]), className="mt-3")
        ]),
        dcc.Tab(label="Quadrant Map", children=[
            dbc.Card(dbc.CardBody([
                html.H5("Quadrant Map Analysis", className="card-title"),
                dcc.RadioItems(
                    id='quadrant-map-type',
                    options=[
                        {'label': 'Key Driver Analysis', 'value': 'stated_vs_derived'},
                        {'label': 'Importance - Performance Analysis', 'value': 'importance_vs_performance'}
                    ],
                    value='stated_vs_derived',
                ),
                html.Div(id='performance-entity-selector-div', children=[
                    html.Hr(),
                    dbc.Row([
                        dbc.Col(html.P("Select an entity to analyze its performance:"), width="auto"),
                        dbc.Col(dcc.Dropdown(id='entity-selector-performance')),
                        dbc.Col(dbc.Button("Download Performance Data", id="btn-download-performance", color="secondary", size="sm"), width="auto")
                    ], align="center"),
                ], style={'display': 'none'}),
                html.Div(id='stated-derived-download-div', children=[
                    html.Hr(),
                    dbc.Row(dbc.Col(dbc.Button("Download Stated vs Derived Data", id="btn-download-stated-derived", color="secondary", size="sm"))),
                ], style={'display': 'none'}),
                html.Hr(),
                dcc.Loading(html.Div(id='quadrant-map-container'))
            ]), className="mt-3")
        ]),
    ])
    return filter_content, tabs
# --- END: REVISED CALLBACK ---


# --- START: NEW CALLBACK TO ENABLE BUTTONS ---
@app.callback(
    Output('btn-download-company', 'disabled'),
    Output('btn-download-app', 'disabled'),
    Input('processed-data-store', 'data')
)
def toggle_download_buttons(json_data):
    if not json_data:
        # If there is no data at all, both buttons are disabled
        return True, True

    # Check if 'company' and 'app' data exist and are not empty in the data store
    company_data_exists = json_data.get('company') and not pd.read_json(io.StringIO(json_data['company']), orient='split').empty
    app_data_exists = json_data.get('app') and not pd.read_json(io.StringIO(json_data['app']), orient='split').empty

    # The button is enabled (disabled=False) if its corresponding data exists
    return not company_data_exists, not app_data_exists
# --- END: NEW CALLBACK ---


@app.callback(
    Output('processed-data-store', 'data'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    prevent_initial_call=True
)
def update_processed_data_store(contents, filename):
    if contents is None: return None
    return process_data(contents, filename)

@app.callback(
    Output('filter-segment', 'options'), Output('filter-region', 'options'), Output('filter-province', 'options'),
    Input('dynamic-filters-placeholder', 'children'), # Triggered when filters are added to the layout
    State('processed-data-store', 'data'),
    State('mode-selector', 'value'),
    prevent_initial_call=True
)
def update_filters_options(filter_children, json_data, mode):
    if not json_data or mode not in json_data: return [], [], []
    df = pd.read_json(io.StringIO(json_data[mode]), orient='split')
    if df.empty: return [], [], []
    seg_opts = [{'label': i, 'value': i} for i in df['Segmen'].unique()]
    reg_opts = [{'label': i, 'value': i} for i in df['Region'].unique()]
    prov_opts = [{'label': i, 'value': i} for i in df['Provinsi'].unique()]
    return seg_opts, reg_opts, prov_opts

@app.callback(
    Output('entity-select', 'options'), Output('entity-select', 'value'),
    Output('entity-select-attribute', 'options'), Output('entity-select-attribute', 'value'),
    Output('entity-selector-performance', 'options'), Output('entity-selector-performance', 'value'),
    Input('tabs-container', 'children'),
    Input('mode-selector', 'value'),
    State('processed-data-store', 'data'),
    State('filter-segment', 'value'),
    State('filter-region', 'value'),
    State('filter-province', 'value'),
    prevent_initial_call=True
)
def update_entity_selectors(tabs_children, mode, json_data, segments, regions, provinces):
    if not json_data or mode not in json_data: return [], [], [], [], [], []
    df = pd.read_json(io.StringIO(json_data[mode]), orient='split')
    df_filtered = filter_dataframe(df, segments, regions, provinces)
    df_filtered['Display_Name'] = df_filtered['Company_Short_Name'].fillna(df_filtered['Original_Name'])
    if df_filtered.empty: return [], [], [], [], [], []
    counts = df_filtered.drop_duplicates(subset=['SbjNum', 'Original_Name']).groupby('Display_Name')['SbjNum'].nunique().sort_values(ascending=False)
    top_3 = counts.head(3).index.tolist()
    all_entities = counts.index.tolist()
    options = [{'label': comp, 'value': comp} for comp in all_entities]
    total_market_option = {'label': 'Total Market', 'value': 'Total Market'}
    options.insert(0, total_market_option)
    default_perf_entity = None
    if mode == 'company':
        if 'Bayer' in all_entities: default_perf_entity = 'Bayer'
    elif mode == 'app':
        bayer_app_name = next((name for name in all_entities if 'Bayer' in name), None)
        if bayer_app_name: default_perf_entity = bayer_app_name
    if not default_perf_entity and all_entities:
        default_perf_entity = all_entities[0]
    return options, top_3, options, top_3, options, default_perf_entity

@app.callback(
    Output('csat-dist-chart', 'figure'),
    Output('nps-dist-chart', 'figure'),
    Output('nps-chart-col', 'style'),
    Input('processed-data-store', 'data'),
    Input('mode-selector', 'value'),
    Input('entity-select', 'value'),
    Input('filter-segment', 'value'),
    Input('filter-region', 'value'),
    Input('filter-province', 'value')
)
def update_kpi_charts(json_data, mode, selected_entities, segments, regions, provinces):
    if not json_data or not selected_entities or mode not in json_data:
        return go.Figure(), go.Figure(), {'display': 'none'}
    df = pd.read_json(io.StringIO(json_data[mode]), orient='split')
    df = filter_dataframe(df, segments, regions, provinces)
    if df.empty:
        return go.Figure(layout={'title': 'No data for current filter selection'}), go.Figure(), {'display': 'none'}
    df['Display_Name'] = df['Company_Short_Name'].fillna(df['Original_Name'])
    csat_col = 'Overall_CSAT_App' if mode == 'app' else 'Overall_CSAT'
    csat_data = []
    unique_person_df = df.drop_duplicates(subset=['SbjNum', 'Original_Name'])
    for entity in selected_entities:
        if entity == 'Total Market':
            filtered_df = unique_person_df
        else:
            filtered_df = unique_person_df[unique_person_df['Display_Name'] == entity]
        n_base = filtered_df['SbjNum'].nunique()
        if n_base == 0: continue
        proportions = filtered_df[csat_col].value_counts(normalize=True).sort_index()
        t2b = proportions.get(9, 0) + proportions.get(10, 0)
        mean_score = filtered_df[csat_col].mean()
        csat_data.append({'Entity': entity, 'N': n_base, 'T2B': t2b, 'Mean': mean_score, 'Proportions': proportions})
    csat_fig = make_subplots(specs=[[{"secondary_y": True}]])
    if csat_data:
        score_order = list(range(1, 11))
        color_gradient = {'1': '#d73027', '2': '#f46d43', '3': '#fdae61', '4': '#fee090', '5': '#ffffbf', '6': '#e0f3f8', '7': '#abd9e9', '8': '#74add1', '9': '#4575b4', '10': '#313695'}
        entities_list = [d['Entity'] for d in csat_data]
        x_axis_labels = [f"<b>{d['Entity']}</b><br>n = {d['N']}<br>T2B: {d['T2B']:.0%}" for d in csat_data]
        for score in score_order:
            csat_fig.add_trace(go.Bar(
                name=f'Score {score}', x=entities_list, y=[d['Proportions'].get(score, 0) for d in csat_data],
                text=[f"{d['Proportions'].get(score, 0)*100:.0f}" for d in csat_data], texttemplate='%{text}', textposition='inside',
                marker_color=color_gradient[str(score)], hovertemplate='<b>%{x}</b><br>Score ' + str(score) + ': %{y:.0%}<extra></extra>'
            ))
        csat_fig.add_trace(go.Scatter(
            name='Mean Score', x=entities_list, y=[d['Mean'] for d in csat_data],
            mode='lines+markers+text', text=[f"{d['Mean']:.2f}" for d in csat_data], textposition='top center',
            marker=dict(color='red', size=8), line=dict(color='red', width=2), yaxis='y2',
            hovertemplate='<b>%{x}</b><br>Mean Score: %{y:.2f}<extra></extra>'
        ))
        csat_fig.update_layout(
            barmode='stack', title_text=f"Overall Satisfaction Distribution ({'App' if mode == 'app' else 'Company'})",
            xaxis=dict(ticktext=x_axis_labels, tickvals=entities_list), yaxis=dict(title='Percentage of Respondents', tickformat='.0%'),
            yaxis2=dict(title='Mean Score', range=[1, 10], overlaying='y', side='right'),
            legend_title_text='Score', legend=dict(traceorder='reversed', orientation='h', yanchor='bottom', y=-0.5, xanchor='center', x=0.5),
            uniformtext_minsize=8, uniformtext_mode='hide', margin=dict(b=150)
        )
    nps_fig = make_subplots(specs=[[{"secondary_y": True}]])
    nps_style = {'display': 'none'}
    if mode == 'company':
        nps_data = []
        for entity in selected_entities:
            if entity == 'Total Market': filtered_df = unique_person_df
            else: filtered_df = unique_person_df[unique_person_df['Display_Name'] == entity]
            n_base = filtered_df['SbjNum'].nunique()
            if n_base == 0: continue
            proportions = filtered_df['NPS_Category'].value_counts(normalize=True)
            promoter_prop = proportions.get('Promoters', 0)
            detractor_prop = proportions.get('Detractors', 0)
            nps_score = (promoter_prop - detractor_prop) * 100
            mean_nps_val = filtered_df['NPS_Score'].mean()
            nps_data.append({'Entity': entity, 'N': n_base, 'NPS_Score': nps_score, 'Mean': mean_nps_val, 'Proportions': proportions})
        if nps_data:
            nps_style = {'display': 'block'}
            category_order = ["Promoters", "Passives", "Detractors"]
            loop_order = category_order[::-1]
            color_map = {"Promoters": "#2ca02c", "Passives": "#ff7f0e", "Detractors": "#d62728"}
            entities_list = [d['Entity'] for d in nps_data]
            x_axis_labels = [f"<b>{d['Entity']}</b><br>n = {d['N']}<br>NPS: {d['NPS_Score']:.0f}" for d in nps_data]
            for cat in loop_order:
                nps_fig.add_trace(go.Bar(
                    name=cat, x=entities_list, y=[d['Proportions'].get(cat, 0) for d in nps_data],
                    text=[f"{d['Proportions'].get(cat, 0)*100:.0f}" for d in nps_data], texttemplate='%{text}', textposition='inside',
                    marker_color=color_map[cat], hovertemplate='<b>%{x}</b><br>' + cat + ': %{y:.0%}<extra></extra>'
                ))
            nps_fig.add_trace(go.Scatter(
                name='Mean Score', x=entities_list, y=[d['Mean'] for d in nps_data],
                mode='lines+markers+text', text=[f"{d['Mean']:.2f}" for d in nps_data], textposition='top center',
                marker=dict(color='red', size=8), line=dict(color='red', width=2), yaxis='y2',
                hovertemplate='<b>%{x}</b><br>Mean Score: %{y:.2f}<extra></extra>'
            ))
            nps_fig.update_layout(
                barmode='stack', title_text="NPS Distribution",
                xaxis=dict(ticktext=x_axis_labels, tickvals=entities_list),
                yaxis=dict(title='Percentage of Respondents', tickformat='.0%'),
                yaxis2=dict(title='Mean Score', range=[0, 10], overlaying='y', side='right'),
                legend_title_text='NPS Category',
                legend=dict(traceorder='reversed', orientation='h', yanchor='bottom', y=-0.5, xanchor='center', x=0.5),
                uniformtext_minsize=8, uniformtext_mode='hide', margin=dict(b=150)
            )
    return csat_fig, nps_fig, nps_style

@app.callback(
    Output('snake-charts-container', 'children'),
    Input('processed-data-store', 'data'), Input('mode-selector', 'value'),
    Input('entity-select-attribute', 'value'),
    Input('filter-segment', 'value'), Input('filter-region', 'value'), Input('filter-province', 'value')
)
def update_snake_charts(json_data, mode, selected_entities, segments, regions, provinces):
    if not json_data or not selected_entities or mode not in json_data: return []
    df = pd.read_json(io.StringIO(json_data[mode]), orient='split')
    df = filter_dataframe(df, segments, regions, provinces)
    if df.empty: return []
    df['Display_Name'] = df['Company_Short_Name'].fillna(df['Original_Name'])
    charts = []
    aspect_names = sorted(df['Aspect_Short_Name'].unique())
    for aspect in aspect_names:
        aspect_data = df[df['Aspect_Short_Name'] == aspect]
        if aspect_data.empty: continue
        attr_order = aspect_data.groupby('Attribute_Short_Name')['Importance_Score'].mean().sort_values(ascending=False).index.tolist()
        attr_perf_data = []
        for entity in selected_entities:
            if entity == 'Total Market': company_df = aspect_data
            else: company_df = aspect_data[aspect_data['Display_Name'] == entity]
            if company_df.empty: continue
            avg_scores = company_df.groupby('Attribute_Short_Name')['Satisfaction_Score'].mean().reindex(attr_order).reset_index()
            avg_scores['Entity'] = entity
            attr_perf_data.append(avg_scores)
        chart_df = pd.concat(attr_perf_data) if attr_perf_data else pd.DataFrame()
        if not chart_df.empty:
            fig = px.line(chart_df, x='Attribute_Short_Name', y='Satisfaction_Score', color='Entity',
                          title=f'Attribute Performance for: {aspect}',
                          labels={'Satisfaction_Score': 'Avg. Score', 'Attribute_Short_Name': 'Attribute (ordered by importance)'},
                          markers=True, category_orders={'Attribute_Short_Name': attr_order})
            fig.update_layout(yaxis_range=[1,10], legend_title_text='Entity')
            charts.append(dcc.Graph(figure=fig))
            charts.append(html.Hr())
    return charts

@app.callback(
    Output('performance-entity-selector-div', 'style'),
    Output('stated-derived-download-div', 'style'),
    Input('quadrant-map-type', 'value')
)
def toggle_quadrant_map_selectors(map_type):
    if map_type == 'importance_vs_performance':
        return {'display': 'block'}, {'display': 'none'}
    elif map_type == 'stated_vs_derived':
        return {'display': 'none'}, {'display': 'block'}
    return {'display': 'none'}, {'display': 'none'}

@app.callback(
    Output('quadrant-map-container', 'children'),
    Input('processed-data-store', 'data'), Input('mode-selector', 'value'),
    Input('quadrant-map-type', 'value'), Input('entity-selector-performance', 'value'),
    Input('filter-segment', 'value'), Input('filter-region', 'value'), Input('filter-province', 'value')
)
def update_quadrant_maps(json_data, mode, map_type, selected_entity, segments, regions, provinces):
    if not json_data or mode not in json_data:
        return html.Div("Please upload data to view this chart.")
    df = pd.read_json(io.StringIO(json_data[mode]), orient='split')
    df['Display_Name'] = df['Company_Short_Name'].fillna(df['Original_Name'])
    df = filter_dataframe(df, segments, regions, provinces)
    if df.empty:
        return html.Div("No data with current filters")

    if map_type == 'stated_vs_derived':
        aspect_names = list(df['Aspect_Short_Name'].unique())
        if mode == 'company':
            aspect_names.insert(0, "Overall")
        charts = []
        for aspect in aspect_names:
            aspect_df_filtered = df[df['Aspect_Short_Name'] == aspect] if aspect != "Overall" else df
            derived_importance_df = calculate_derived_importance(df, mode, aspect_name=aspect if aspect != "Overall" else None)
            if derived_importance_df.empty:
                charts.append(html.H4(f"Key Driver Analysis for: {aspect}"))
                charts.append(html.P("Not enough data to calculate Derived Importance."))
                charts.append(html.Hr())
                continue
            stated_importance_df = aspect_df_filtered.groupby('Attribute_Short_Name')['Importance_Score'].mean().reset_index()
            stated_importance_df.rename(columns={'Attribute_Short_Name': 'feature', 'Importance_Score': 'Stated_Importance'}, inplace=True)
            quadrant_df = pd.merge(stated_importance_df, derived_importance_df, on='feature')
            if quadrant_df.empty: continue
            x_col, y_col = 'Derived_Importance', 'Stated_Importance'
            title = f'Key Driver Analysis for: {aspect}'
            padding = 0.5
            min_x, max_x = quadrant_df[x_col].min() - padding, quadrant_df[x_col].max() + padding
            min_y, max_y = quadrant_df[y_col].min() - padding, quadrant_df[y_col].max() + padding
            mean_x, mean_y = quadrant_df[x_col].mean(), quadrant_df[y_col].mean()
            fig = px.scatter(quadrant_df, x=x_col, y=y_col, text='feature', title=title)
            fig.update_traces(textposition='top center', textfont_size=10)
            fig.update_layout(
                height=600, xaxis_title="Derived Importance", yaxis_title="Stated Importance",
                xaxis=dict(range=[min_x, max_x], showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(range=[min_y, max_y], showgrid=False, zeroline=False, showticklabels=False),
                shapes=[
                    dict(type="rect", xref="x", yref="y", x0=mean_x, y0=mean_y, x1=max_x, y1=max_y, fillcolor="PaleGreen", opacity=0.3, layer="below", line_width=0),
                    dict(type="rect", xref="x", yref="y", x0=min_x, y0=mean_y, x1=mean_x, y1=max_y, fillcolor="LightSkyBlue", opacity=0.3, layer="below", line_width=0),
                    dict(type="rect", xref="x", yref="y", x0=mean_x, y0=min_y, x1=max_x, y1=mean_y, fillcolor="SandyBrown", opacity=0.3, layer="below", line_width=0),
                    dict(type="rect", xref="x", yref="y", x0=min_x, y0=min_y, x1=mean_x, y1=mean_y, fillcolor="LightGray", opacity=0.3, layer="below", line_width=0),
                ]
            )
            fig.add_annotation(x=max_x, y=max_y, text="Key Drivers", showarrow=False, xanchor='right', yanchor='top', font=dict(color="DarkGreen"))
            fig.add_annotation(x=min_x, y=max_y, text="Stated Drivers", showarrow=False, xanchor='left', yanchor='top', font=dict(color="DarkBlue"))
            fig.add_annotation(x=max_x, y=min_y, text="Basic Drivers", showarrow=False, xanchor='right', yanchor='bottom', font=dict(color="DarkOrange"))
            fig.add_annotation(x=min_x, y=min_y, text="Less Influential", showarrow=False, xanchor='left', yanchor='bottom', font=dict(color="DimGray"))
            fig.add_vline(x=mean_x, line_dash="dash", line_color="grey")
            fig.add_hline(y=mean_y, line_dash="dash", line_color="grey")
            charts.append(dcc.Graph(figure=fig))
            charts.append(html.Hr())
        return html.Div(charts)

    elif map_type == 'importance_vs_performance':
        if not selected_entity:
            return html.Div("Please select an entity to generate the performance map.")
        derived_importance_df = calculate_derived_importance(df, mode, aspect_name=None)
        if derived_importance_df.empty:
            return html.P("Not enough data to calculate Derived Importance.")
        perf_market = df.groupby('Attribute_Short_Name')['Satisfaction_Score'].mean()
        if selected_entity == 'Total Market':
            performance_df = perf_market.reset_index()
            performance_df.columns = ['feature', 'Performance']
            x_line = perf_market.mean()
        else:
            entity_df = df[df['Display_Name'] == selected_entity]
            if entity_df.empty:
                return html.P(f"No data found for the selected entity: {selected_entity}")
            perf_entity = entity_df.groupby('Attribute_Short_Name')['Satisfaction_Score'].mean()
            stdev_entity = entity_df.groupby('Attribute_Short_Name')['Satisfaction_Score'].std()
            perf_market_aligned = perf_market.reindex(perf_entity.index)
            z_score = (perf_entity - perf_market_aligned) / stdev_entity
            performance_df = z_score.reset_index()
            performance_df.columns = ['feature', 'Performance']
            x_line = 0
        quadrant_df = pd.merge(performance_df, derived_importance_df, on='feature')
        quadrant_df.dropna(inplace=True)
        if quadrant_df.empty:
            return html.P("Not enough data to plot the performance map.")
        x_col, y_col = 'Performance', 'Derived_Importance'
        title = f'Importance - Performance Analysis for: {selected_entity}'
        if selected_entity != 'Total Market':
            abs_max = quadrant_df[x_col].abs().max() * 1.1
            min_x, max_x = -abs_max, abs_max
        else:
            padding_x = 0.5
            min_x, max_x = quadrant_df[x_col].min() - padding_x, quadrant_df[x_col].max() + padding_x
        padding_y = 0.5
        min_y, max_y = quadrant_df[y_col].min() - padding_y, quadrant_df[y_col].max() + padding_y
        mean_y = quadrant_df[y_col].mean()
        fig = px.scatter(quadrant_df, x=x_col, y=y_col, text='feature', title=title)
        fig.update_traces(textposition='top center', textfont_size=10)
        fig.update_layout(
            height=700, xaxis_title="Performance (vs Market Avg)", yaxis_title="Derived Importance",
            xaxis=dict(range=[min_x, max_x], showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(range=[min_y, max_y], showgrid=False, zeroline=False, showticklabels=False),
            shapes=[
                dict(type="rect", xref="x", yref="y", x0=x_line, y0=mean_y, x1=max_x, y1=max_y, fillcolor="#BADDD3", opacity=0.4, layer="below", line_width=0),
                dict(type="rect", xref="x", yref="y", x0=min_x, y0=mean_y, x1=x_line, y1=max_y, fillcolor="#C5D3E3", opacity=0.4, layer="below", line_width=0),
                dict(type="rect", xref="x", yref="y", x0=x_line, y0=min_y, x1=max_x, y1=mean_y, fillcolor="#F5DDB3", opacity=0.4, layer="below", line_width=0),
                dict(type="rect", xref="x", yref="y", x0=min_x, y0=min_y, x1=x_line, y1=mean_y, fillcolor="#DDDDDD", opacity=0.4, layer="below", line_width=0),
            ]
        )
        fig.add_annotation(x=max_x, y=max_y, text="Key Strengths", showarrow=False, xanchor='right', yanchor='top', font=dict(color="DarkGreen"))
        fig.add_annotation(x=min_x, y=max_y, text="Area for Improvement", showarrow=False, xanchor='left', yanchor='top', font=dict(color="DarkBlue"))
        fig.add_annotation(x=max_x, y=min_y, text="Basic Benefit", showarrow=False, xanchor='right', yanchor='bottom', font=dict(color="DarkOrange"))
        fig.add_annotation(x=min_x, y=min_y, text="Less Relevant", showarrow=False, xanchor='left', yanchor='bottom', font=dict(color="DimGray"))
        fig.add_vline(x=x_line, line_dash="dash", line_color="grey")
        fig.add_hline(y=mean_y, line_dash="dash", line_color="grey")
        return dcc.Graph(figure=fig)

@app.callback(
    Output("download-kpi-data", "data"),
    Input("btn-download-kpi", "n_clicks"),
    State('processed-data-store', 'data'), State('mode-selector', 'value'),
    State('entity-select', 'value'), State('filter-segment', 'value'),
    State('filter-region', 'value'), State('filter-province', 'value'),
    prevent_initial_call=True,
)
def download_kpi_data(n_clicks, json_data, mode, selected_entities, segments, regions, provinces):
    if not n_clicks or not selected_entities: return None
    df = pd.read_json(io.StringIO(json_data[mode]), orient='split')
    df_filtered = filter_dataframe(df, segments, regions, provinces)
    df_filtered['Display_Name'] = df_filtered['Company_Short_Name'].fillna(df_filtered['Original_Name'])
    output_rows = []
    unique_person_df = df_filtered.drop_duplicates(subset=['SbjNum', 'Original_Name'])
    for entity in selected_entities:
        if entity == 'Total Market': entity_df = unique_person_df
        else: entity_df = unique_person_df[unique_person_df['Display_Name'] == entity]
        n_base = entity_df['SbjNum'].nunique()
        if n_base == 0: continue
        csat_col = 'Overall_CSAT_App' if mode == 'app' else 'Overall_CSAT'
        mean_csat = entity_df[csat_col].mean()
        proportions_csat = entity_df[csat_col].value_counts(normalize=True)
        t2b = proportions_csat.get(9, 0) + proportions_csat.get(10, 0)
        row = {'Entity': entity, 'Base N': n_base, 'Overall Satisfaction Mean': mean_csat, 'T2B (9+10) %': t2b}
        for i in range(1, 11):
            row[f'CSAT {i} %'] = proportions_csat.get(i, 0)
        if mode == 'company':
            mean_nps = entity_df['NPS_Score'].mean()
            proportions_nps = entity_df['NPS_Category'].value_counts(normalize=True)
            promoter_prop = proportions_nps.get('Promoters', 0)
            detractor_prop = proportions_nps.get('Detractors', 0)
            nps_score = (promoter_prop - detractor_prop) * 100
            row.update({
                'NPS Score': nps_score, 'NPS Mean': mean_nps,
                'Promoters %': promoter_prop, 'Passives %': proportions_nps.get('Passives', 0),
                'Detractors %': detractor_prop
            })
        output_rows.append(row)
    output_df = pd.DataFrame(output_rows)
    return dcc.send_data_frame(output_df.to_excel, "kpi_summary_data.xlsx", sheet_name="KPI_Summary", index=False)

@app.callback(
    Output("download-attribute-data", "data"),
    Input("btn-download-attribute", "n_clicks"),
    State('processed-data-store', 'data'), State('mode-selector', 'value'),
    State('entity-select-attribute', 'value'), State('filter-segment', 'value'),
    State('filter-region', 'value'), State('filter-province', 'value'),
    prevent_initial_call=True,
)
def download_attribute_data(n_clicks, json_data, mode, selected_entities, segments, regions, provinces):
    if not n_clicks or not selected_entities: return None
    df = pd.read_json(io.StringIO(json_data[mode]), orient='split')
    df_filtered = filter_dataframe(df, segments, regions, provinces)
    df_filtered['Display_Name'] = df_filtered['Company_Short_Name'].fillna(df_filtered['Original_Name'])
    output_rows = []
    aspect_names = sorted(df_filtered['Aspect_Short_Name'].unique())
    for aspect in aspect_names:
        aspect_data = df_filtered[df_filtered['Aspect_Short_Name'] == aspect]
        for entity in selected_entities:
            if entity == 'Total Market': entity_df = aspect_data
            else: entity_df = aspect_data[aspect_data['Display_Name'] == entity]
            if entity_df.empty: continue
            summary = entity_df.groupby('Attribute_Short_Name').agg(
                Avg_Satisfaction_Score=('Satisfaction_Score', 'mean'),
                Stated_Importance_Score=('Importance_Score', 'mean')
            ).reset_index()
            summary['Entity'] = entity
            summary['Aspect'] = aspect
            output_rows.append(summary)
    output_df = pd.concat(output_rows, ignore_index=True)
    output_df = output_df[['Aspect', 'Attribute_Short_Name', 'Entity', 'Avg_Satisfaction_Score', 'Stated_Importance_Score']]
    return dcc.send_data_frame(output_df.to_excel, "attribute_summary_data.xlsx", sheet_name="Attribute_Summary", index=False)

@app.callback(
    Output("download-stated-derived-data", "data"),
    Input("btn-download-stated-derived", "n_clicks"),
    State('processed-data-store', 'data'), State('mode-selector', 'value'),
    State('filter-segment', 'value'), State('filter-region', 'value'), State('filter-province', 'value'),
    prevent_initial_call=True,
)
def download_stated_derived_data(n_clicks, json_data, mode, segments, regions, provinces):
    if not n_clicks: return None
    df = pd.read_json(io.StringIO(json_data[mode]), orient='split')
    df_filtered = filter_dataframe(df, segments, regions, provinces)
    all_quadrant_data = []
    aspect_names = list(df_filtered['Aspect_Short_Name'].unique())
    if mode == 'company': aspect_names.insert(0, "Overall")
    for aspect in aspect_names:
        aspect_df = df_filtered[df_filtered['Aspect_Short_Name'] == aspect] if aspect != "Overall" else df_filtered
        if aspect_df.empty: continue
        derived_importance_df = calculate_derived_importance(df_filtered, mode, aspect_name=aspect if aspect != "Overall" else None)
        if derived_importance_df.empty: continue
        stated_importance_df = aspect_df.groupby('Attribute_Short_Name')['Importance_Score'].mean().reset_index()
        stated_importance_df.rename(columns={'Attribute_Short_Name': 'feature', 'Importance_Score': 'Stated_Importance'}, inplace=True)
        quadrant_df = pd.merge(stated_importance_df, derived_importance_df, on='feature')
        quadrant_df['Aspect'] = aspect
        all_quadrant_data.append(quadrant_df)
    if not all_quadrant_data: return None
    output_df = pd.concat(all_quadrant_data, ignore_index=True)
    output_df = output_df[['Aspect', 'feature', 'Stated_Importance', 'Derived_Importance']]
    output_df.rename(columns={'feature': 'Attribute'}, inplace=True)
    return dcc.send_data_frame(output_df.to_excel, "stated_vs_derived_data.xlsx", sheet_name="Stated_v_Derived", index=False)

@app.callback(
    Output("download-performance-data-xlsx", "data"),
    Input("btn-download-performance", "n_clicks"),
    State('processed-data-store', 'data'), State('mode-selector', 'value'),
    State('entity-selector-performance', 'value'), State('filter-segment', 'value'),
    State('filter-region', 'value'), State('filter-province', 'value'),
    prevent_initial_call=True,
)
def download_performance_data(n_clicks, json_data, mode, selected_entity, segments, regions, provinces):
    if not n_clicks or not selected_entity: return None
    df = pd.read_json(io.StringIO(json_data[mode]), orient='split')
    df['Display_Name'] = df['Company_Short_Name'].fillna(df['Original_Name'])
    df = filter_dataframe(df, segments, regions, provinces)
    derived_importance_df = calculate_derived_importance(df, mode, aspect_name=None)
    perf_market = df.groupby('Attribute_Short_Name')['Satisfaction_Score'].mean()
    entity_df = df[df['Display_Name'] == selected_entity]
    perf_entity = entity_df.groupby('Attribute_Short_Name')['Satisfaction_Score'].mean()
    stdev_entity = entity_df.groupby('Attribute_Short_Name')['Satisfaction_Score'].std()
    output_df = derived_importance_df.rename(columns={'feature': 'Attribute'})
    output_df = pd.merge(output_df, perf_entity.rename('Absolute_Performance_Entity'), left_on='Attribute', right_index=True, how='left')
    output_df = pd.merge(output_df, perf_market.rename('Absolute_Performance_Market'), left_on='Attribute', right_index=True, how='left')
    output_df = pd.merge(output_df, stdev_entity.rename('Stdev_Entity'), left_on='Attribute', right_index=True, how='left')
    filename = f"performance_data_{selected_entity.replace(' ', '_')}.xlsx"
    return dcc.send_data_frame(output_df.to_excel, filename, sheet_name="Performance_Calc", index=False)


@app.callback(
    Output("download-dataframe-xlsx", "data"),
    Input("btn-download-company", "n_clicks"), Input("btn-download-app", "n_clicks"),
    State('processed-data-store', 'data'),
    prevent_initial_call=True,
)
def download_long_data(n_clicks_company, n_clicks_app, json_data):
    if not ctx.triggered_id or not json_data: return None
    button_id = ctx.triggered_id
    if button_id == "btn-download-company" and 'company' in json_data:
        df = pd.read_json(io.StringIO(json_data['company']), orient='split')
        if not df.empty: return dcc.send_data_frame(df.to_excel, "processed_company_data_long.xlsx", sheet_name="Processed_Data", index=False)
    elif button_id == "btn-download-app" and 'app' in json_data:
        df = pd.read_json(io.StringIO(json_data['app']), orient='split')
        if not df.empty: return dcc.send_data_frame(df.to_excel, "processed_app_data_long.xlsx", sheet_name="Processed_Data", index=False)
    return None

if __name__ == '__main__':
    app.run(debug=True)