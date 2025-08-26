pip install dash pandas plotly
import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px

# Load dataset
df = px.data.gapminder()

# Initialize app
app = dash.Dash(__name__)
app.title = "Gapminder Dashboard"

# Layout
app.layout = html.Div([
    html.H1("Gapminder Interactive Dashboard", style={'textAlign': 'center'}),

    html.Label("Select Year:"),
    dcc.Slider(
        id='year-slider',
        min=df['year'].min(),
        max=df['year'].max(),
        step=5,
        marks={str(year): str(year) for year in df['year'].unique()},
        value=df['year'].min()
    ),

    html.Label("Select Continent:"),
    dcc.Dropdown(
        id='continent-dropdown',
        options=[{'label': c, 'value': c} for c in df['continent'].unique()],
        value='Asia',
        multi=False
    ),

    dcc.Graph(id='scatter-plot'),

    html.Div(id='summary', style={'marginTop': 30})
])

# Callback for interactivity
@app.callback(
    [Output('scatter-plot', 'figure'),
     Output('summary', 'children')],
    [Input('year-slider', 'value'),
     Input('continent-dropdown', 'value')]
)
def update_figure(selected_year, selected_continent):
    filtered_df = df[(df.year == selected_year) & (df.continent == selected_continent)]

    fig = px.scatter(
        filtered_df,
        x="gdpPercap",
        y="lifeExp",
        size="pop",
        color="country",
        hover_name="country",
        log_x=True,
        size_max=60,
        title=f"{selected_continent} Countries in {selected_year}"
    )

    summary = f"Showing {len(filtered_df)} countries from {selected_continent} in {selected_year}."

    return fig, summary

# Run the app
if __name__ == '__main__':
    app.run(debug=True)