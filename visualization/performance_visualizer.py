"""
Module pour la visualisation des performances des stratégies obligataires.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class PerformanceVisualizer:
    """
    Classe pour visualiser les performances des stratégies obligataires.
    """
    
    def __init__(self, style='seaborn', figsize=(12, 8)):
        """
        Initialise le visualiseur de performances.
        
        Args:
            style (str, optional): Style de matplotlib à utiliser. Par défaut 'seaborn'.
            figsize (tuple, optional): Taille par défaut des figures. Par défaut (12, 8).
        """
        plt.style.use(style)
        self.figsize = figsize
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
    def plot_portfolio_value(self, strategies_data, benchmark_data=None, log_scale=False):
        """
        Trace l'évolution de la valeur du portefeuille pour différentes stratégies.
        
        Args:
            strategies_data (dict): Dictionnaire {nom_stratégie: DataFrame} avec les données de chaque stratégie.
            benchmark_data (pandas.DataFrame, optional): Données de l'indice de référence. Par défaut None.
            log_scale (bool, optional): Utiliser une échelle logarithmique. Par défaut False.
            
        Returns:
            matplotlib.figure.Figure: Figure contenant le graphique.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Tracer les stratégies
        for i, (name, data) in enumerate(strategies_data.items()):
            if 'value' in data.columns:
                ax.plot(data.index, data['value'], label=name, color=self.colors[i % len(self.colors)])
                
        # Ajouter l'indice de référence si fourni
        if benchmark_data is not None and 'value' in benchmark_data.columns:
            ax.plot(benchmark_data.index, benchmark_data['value'], label='Benchmark', 
                    color='black', linestyle='--', linewidth=2)
                
        ax.set_title('Évolution de la Valeur du Portefeuille')
        ax.set_xlabel('Date')
        ax.set_ylabel('Valeur ($)')
        
        # Appliquer l'échelle logarithmique si demandé
        if log_scale:
            ax.set_yscale('log')
            
        # Formater l'axe des X pour les dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        plt.xticks(rotation=45)
        
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        
        return fig
    
    def plot_returns_comparison(self, strategies_data, benchmark_data=None, period='M'):
        """
        Compare les rendements périodiques des différentes stratégies.
        
        Args:
            strategies_data (dict): Dictionnaire {nom_stratégie: DataFrame} avec les données de chaque stratégie.
            benchmark_data (pandas.DataFrame, optional): Données de l'indice de référence. Par défaut None.
            period (str, optional): Période de rééchantillonnage ('D', 'W', 'M', 'Q', 'Y'). Par défaut 'M'.
            
        Returns:
            matplotlib.figure.Figure: Figure contenant le graphique.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Calculer les rendements périodiques pour chaque stratégie
        periodic_returns = {}
        
        for name, data in strategies_data.items():
            if 'returns' in data.columns:
                # Rééchantillonner les rendements
                if period == 'D':
                    periodic_returns[name] = data['returns']
                else:
                    # Pour les périodes plus longues, calculer le rendement composé
                    periodic_returns[name] = (1 + data['returns']).resample(period).prod() - 1
                    
        # Ajouter l'indice de référence si fourni
        if benchmark_data is not None and 'returns' in benchmark_data.columns:
            if period == 'D':
                periodic_returns['Benchmark'] = benchmark_data['returns']
            else:
                periodic_returns['Benchmark'] = (1 + benchmark_data['returns']).resample(period).prod() - 1
                
        # Créer un DataFrame avec tous les rendements
        returns_df = pd.DataFrame(periodic_returns)
        
        # Tracer les rendements
        returns_df.plot(kind='bar', ax=ax)
        
        # Définir les titres et étiquettes
        period_labels = {
            'D': 'Journaliers',
            'W': 'Hebdomadaires',
            'M': 'Mensuels',
            'Q': 'Trimestriels',
            'Y': 'Annuels'
        }
        
        ax.set_title(f'Rendements {period_labels.get(period, "")} des Stratégies')
        ax.set_xlabel('Date')
        ax.set_ylabel(f'Rendement')
        
        ax.legend()
        ax.grid(True, axis='y')
        plt.tight_layout()
        
        return fig
    
    def plot_drawdown(self, strategies_data, benchmark_data=None):
        """
        Trace les drawdowns pour différentes stratégies.
        
        Args:
            strategies_data (dict): Dictionnaire {nom_stratégie: DataFrame} avec les données de chaque stratégie.
            benchmark_data (pandas.DataFrame, optional): Données de l'indice de référence. Par défaut None.
            
        Returns:
            matplotlib.figure.Figure: Figure contenant le graphique.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Tracer les drawdowns pour chaque stratégie
        for i, (name, data) in enumerate(strategies_data.items()):
            if 'drawdown' in data.columns:
                ax.fill_between(
                    data.index, 
                    0, 
                    data['drawdown'] * 100,  # Convertir en pourcentage
                    alpha=0.3, 
                    color=self.colors[i % len(self.colors)],
                    label=name
                )
                
        # Ajouter l'indice de référence si fourni
        if benchmark_data is not None and 'drawdown' in benchmark_data.columns:
            ax.fill_between(
                benchmark_data.index, 
                0, 
                benchmark_data['drawdown'] * 100,  # Convertir en pourcentage
                alpha=0.3, 
                color='black',
                label='Benchmark'
            )
            
        ax.set_title('Drawdowns des Stratégies (%)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        
        # Formater l'axe des X pour les dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        plt.xticks(rotation=45)
        
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        
        return fig
    
    def plot_rolling_metrics(self, strategies_data, benchmark_data=None, metric='returns', window=252):
        """
        Trace les métriques glissantes pour différentes stratégies.
        
        Args:
            strategies_data (dict): Dictionnaire {nom_stratégie: DataFrame} avec les données de chaque stratégie.
            benchmark_data (pandas.DataFrame, optional): Données de l'indice de référence. Par défaut None.
            metric (str, optional): Métrique à tracer ('returns', 'volatility', 'sharpe'). Par défaut 'returns'.
            window (int, optional): Taille de la fenêtre glissante en jours. Par défaut 252 (1 an).
            
        Returns:
            matplotlib.figure.Figure: Figure contenant le graphique.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Calculer les métriques glissantes pour chaque stratégie
        for i, (name, data) in enumerate(strategies_data.items()):
            if 'returns' in data.columns:
                returns = data['returns'].dropna()
                
                if metric == 'returns':
                    # Rendements annualisés
                    rolling_metric = returns.rolling(window=window).mean() * 252 * 100  # En pourcentage
                    ylabel = 'Rendement Annualisé (%)'
                elif metric == 'volatility':
                    # Volatilité annualisée
                    rolling_metric = returns.rolling(window=window).std() * np.sqrt(252) * 100  # En pourcentage
                    ylabel = 'Volatilité Annualisée (%)'
                elif metric == 'sharpe':
                    # Ratio de Sharpe
                    rolling_returns = returns.rolling(window=window).mean() * 252
                    rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
                    rolling_metric = (rolling_returns - 0.02) / rolling_vol  # Taux sans risque de 2%
                    ylabel = 'Ratio de Sharpe'
                else:
                    raise ValueError(f"Unknown metric '{metric}'. Use 'returns', 'volatility', or 'sharpe'.")
                    
                ax.plot(rolling_metric.index, rolling_metric, label=name, color=self.colors[i % len(self.colors)])
                
        # Ajouter l'indice de référence si fourni
        if benchmark_data is not None and 'returns' in benchmark_data.columns:
            returns = benchmark_data['returns'].dropna()
            
            if metric == 'returns':
                rolling_metric = returns.rolling(window=window).mean() * 252 * 100
            elif metric == 'volatility':
                rolling_metric = returns.rolling(window=window).std() * np.sqrt(252) * 100
            elif metric == 'sharpe':
                rolling_returns = returns.rolling(window=window).mean() * 252
                rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
                rolling_metric = (rolling_returns - 0.02) / rolling_vol
                
            ax.plot(rolling_metric.index, rolling_metric, label='Benchmark', 
                    color='black', linestyle='--', linewidth=2)
                
        # Définir les titres et étiquettes
        metric_labels = {
            'returns': 'Rendements',
            'volatility': 'Volatilité',
            'sharpe': 'Ratio de Sharpe'
        }
        
        ax.set_title(f'{metric_labels.get(metric, metric.capitalize())} Glissant sur {window} jours')
        ax.set_xlabel('Date')
        ax.set_ylabel(ylabel)
        
        # Formater l'axe des X pour les dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        plt.xticks(rotation=45)
        
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        
        return fig
    
    def plot_performance_heatmap(self, strategies_data, metric='returns', period='M'):
        """
        Crée une heatmap des performances périodiques des stratégies.
        
        Args:
            strategies_data (dict): Dictionnaire {nom_stratégie: DataFrame} avec les données de chaque stratégie.
            metric (str, optional): Métrique à tracer ('returns', 'volatility', 'sharpe'). Par défaut 'returns'.
            period (str, optional): Période de rééchantillonnage ('W', 'M', 'Q', 'Y'). Par défaut 'M'.
            
        Returns:
            matplotlib.figure.Figure: Figure contenant la heatmap.
        """
        # Calculer les métriques périodiques pour chaque stratégie
        periodic_data = {}
        
        for name, data in strategies_data.items():
            if 'returns' in data.columns:
                returns = data['returns'].dropna()
                
                # Rééchantillonner les données
                if period == 'M':
                    grouper = pd.Grouper(freq='M')
                elif period == 'Q':
                    grouper = pd.Grouper(freq='Q')
                elif period == 'Y':
                    grouper = pd.Grouper(freq='Y')
                elif period == 'W':
                    grouper = pd.Grouper(freq='W')
                else:
                    raise ValueError(f"Unknown period '{period}'. Use 'W', 'M', 'Q', or 'Y'.")
                    
                grouped_returns = returns.groupby(grouper)
                
                if metric == 'returns':
                    # Rendements composés
                    periodic_data[name] = (1 + grouped_returns).prod() - 1
                elif metric == 'volatility':
                    # Volatilité
                    periodic_data[name] = grouped_returns.std()
                elif metric == 'sharpe':
                    # Ratio de Sharpe
                    mean_returns = grouped_returns.mean()
                    std_returns = grouped_returns.std()
                    periodic_data[name] = (mean_returns - 0.02 / 252) / std_returns  # Ajuster le taux sans risque
                else:
                    raise ValueError(f"Unknown metric '{metric}'. Use 'returns', 'volatility', or 'sharpe'.")
                    
        # Créer un DataFrame avec toutes les données
        df = pd.DataFrame(periodic_data)
        
        # Convertir en pourcentage si nécessaire
        if metric == 'returns' or metric == 'volatility':
            df = df * 100
            
        # Créer la heatmap
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Définir la palette de couleurs
        if metric == 'returns' or metric == 'sharpe':
            cmap = 'RdYlGn'  # Rouge pour négatif, vert pour positif
            center = 0
        else:  # volatility
            cmap = 'YlOrRd'  # Jaune pour faible, rouge pour élevé
            center = None
            
        sns.heatmap(df, annot=True, fmt='.2f', cmap=cmap, center=center, ax=ax)
        
        # Définir les titres et étiquettes
        metric_labels = {
            'returns': 'Rendements (%)',
            'volatility': 'Volatilité (%)',
            'sharpe': 'Ratio de Sharpe'
        }
        
        period_labels = {
            'W': 'Hebdomadaires',
            'M': 'Mensuels',
            'Q': 'Trimestriels',
            'Y': 'Annuels'
        }
        
        ax.set_title(f'{metric_labels.get(metric, metric.capitalize())} {period_labels.get(period, "")} des Stratégies')
        
        plt.tight_layout()
        
        return fig
    
    def plot_performance_radar(self, performance_metrics, metrics=None):
        """
        Crée un graphique radar des métriques de performance pour différentes stratégies.
        
        Args:
            performance_metrics (pandas.DataFrame): DataFrame avec les métriques de performance par stratégie.
            metrics (list, optional): Liste des métriques à inclure. Par défaut None (toutes).
            
        Returns:
            plotly.graph_objects.Figure: Figure Plotly contenant le graphique radar.
        """
        if metrics is None:
            metrics = performance_metrics.columns.tolist()
            
        # Ne garder que les métriques numériques
        numeric_metrics = performance_metrics[metrics].select_dtypes(include=[np.number]).columns.tolist()
        
        # Normaliser les métriques pour une meilleure visualisation
        normalized_metrics = pd.DataFrame(index=performance_metrics.index)
        
        for metric in numeric_metrics:
            # Pour max_drawdown, une valeur plus élevée (moins négative) est meilleure
            if metric == 'max_drawdown':
                values = performance_metrics[metric]
                min_val = values.min()
                max_val = values.max()
                if min_val != max_val:
                    normalized_metrics[metric] = 1 - (values - min_val) / (max_val - min_val)
                else:
                    normalized_metrics[metric] = 1
            else:
                values = performance_metrics[metric]
                min_val = values.min()
                max_val = values.max()
                if min_val != max_val:
                    normalized_metrics[metric] = (values - min_val) / (max_val - min_val)
                else:
                    normalized_metrics[metric] = 1
                    
        # Créer le graphique radar
        fig = go.Figure()
        
        for strategy in normalized_metrics.index:
            fig.add_trace(go.Scatterpolar(
                r=normalized_metrics.loc[strategy].values,
                theta=numeric_metrics,
                fill='toself',
                name=strategy
            ))
            
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title='Comparaison des Métriques de Performance',
            showlegend=True
        )
        
        return fig
    
    def plot_interactive_performance(self, strategies_data, benchmark_data=None):
        """
        Crée un dashboard interactif des performances avec Plotly.
        
        Args:
            strategies_data (dict): Dictionnaire {nom_stratégie: DataFrame} avec les données de chaque stratégie.
            benchmark_data (pandas.DataFrame, optional): Données de l'indice de référence. Par défaut None.
            
        Returns:
            plotly.graph_objects.Figure: Figure Plotly contenant le dashboard.
        """
        # Créer un subplot avec 2 rangées, 2 colonnes
        fig = make_subplots(
            rows=2, 
            cols=2,
            subplot_titles=(
                'Valeur du Portefeuille', 
                'Drawdowns', 
                'Rendements Cumulatifs', 
                'Rendements Mensuels'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "bar"}]
            ]
        )
        
        # Couleurs pour les stratégies
        colors = px.colors.qualitative.Plotly
        
        # 1. Valeur du portefeuille
        for i, (name, data) in enumerate(strategies_data.items()):
            if 'value' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index, 
                        y=data['value'], 
                        name=name,
                        line=dict(color=colors[i % len(colors)]),
                        legendgroup=name
                    ),
                    row=1, col=1
                )
                
        # Ajouter l'indice de référence
        if benchmark_data is not None and 'value' in benchmark_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=benchmark_data.index, 
                    y=benchmark_data['value'], 
                    name='Benchmark',
                    line=dict(color='black', dash='dash'),
                    legendgroup='Benchmark'
                ),
                row=1, col=1
            )
            
        # 2. Drawdowns
        for i, (name, data) in enumerate(strategies_data.items()):
            if 'drawdown' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index, 
                        y=data['drawdown'] * 100,  # Convertir en pourcentage
                        name=name,
                        line=dict(color=colors[i % len(colors)]),
                        legendgroup=name,
                        showlegend=False
                    ),
                    row=1, col=2
                )
                
        # Ajouter l'indice de référence
        if benchmark_data is not None and 'drawdown' in benchmark_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=benchmark_data.index, 
                    y=benchmark_data['drawdown'] * 100,  # Convertir en pourcentage
                    name='Benchmark',
                    line=dict(color='black', dash='dash'),
                    legendgroup='Benchmark',
                    showlegend=False
                ),
                row=1, col=2
            )
            
        # 3. Rendements cumulatifs
        for i, (name, data) in enumerate(strategies_data.items()):
            if 'cumulative_return' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index, 
                        y=data['cumulative_return'],
                        name=name,
                        line=dict(color=colors[i % len(colors)]),
                        legendgroup=name,
                        showlegend=False
                    ),
                    row=2, col=1
                )
                
        # Ajouter l'indice de référence
        if benchmark_data is not None and 'cumulative_return' in benchmark_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=benchmark_data.index, 
                    y=benchmark_data['cumulative_return'],
                    name='Benchmark',
                    line=dict(color='black', dash='dash'),
                    legendgroup='Benchmark',
                    showlegend=False
                ),
                row=2, col=1
            )
            
        # 4. Rendements mensuels
        # Préparer les données de rendements mensuels
        monthly_returns = {}
        
        for name, data in strategies_data.items():
            if 'returns' in data.columns:
                # Calculer les rendements mensuels
                monthly_returns[name] = (1 + data['returns']).resample('M').prod() - 1
                
        # Ajouter l'indice de référence
        if benchmark_data is not None and 'returns' in benchmark_data.columns:
            monthly_returns['Benchmark'] = (1 + benchmark_data['returns']).resample('M').prod() - 1
            
        # Créer un DataFrame avec tous les rendements mensuels
        monthly_df = pd.DataFrame(monthly_returns)
        
        # Créer un graphique à barres pour chaque stratégie
        for i, name in enumerate(monthly_df.columns):
            fig.add_trace(
                go.Bar(
                    x=monthly_df.index, 
                    y=monthly_df[name] * 100,  # Convertir en pourcentage
                    name=name,
                    marker_color=colors[i % len(colors)] if name != 'Benchmark' else 'black',
                    legendgroup=name,
                    showlegend=False
                ),
                row=2, col=2
            )
            
        # Mettre à jour les axes et les titres
        fig.update_layout(
            title='Dashboard de Performance des Stratégies Obligataires',
            height=800,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )
        
        # Mettre à jour les axes
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=2)
        
        fig.update_yaxes(title_text="Valeur ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=1, col=2)
        fig.update_yaxes(title_text="Rendement Cumulatif", row=2, col=1)
        fig.update_yaxes(title_text="Rendement Mensuel (%)", row=2, col=2)
        
        return fig
