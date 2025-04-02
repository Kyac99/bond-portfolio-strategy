"""
Module pour le backtest des stratégies obligataires.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class BacktestEngine:
    """
    Moteur de backtest pour comparer les performances de différentes stratégies obligataires.
    """
    
    def __init__(self, data_loader, strategies=None, benchmark_ticker="AGG"):
        """
        Initialise le moteur de backtest.
        
        Args:
            data_loader: Instance de DataLoader pour charger les données.
            strategies (list, optional): Liste des stratégies à tester. Par défaut None.
            benchmark_ticker (str, optional): Ticker de l'indice de référence. Par défaut "AGG".
        """
        self.data_loader = data_loader
        self.strategies = strategies if strategies else []
        self.benchmark_ticker = benchmark_ticker
        self.results = {}
        self.performance_metrics = {}
        
    def add_strategy(self, strategy):
        """
        Ajoute une stratégie au backtest.
        
        Args:
            strategy: Instance d'une stratégie obligataire.
        """
        self.strategies.append(strategy)
        
    def run_backtest(self, start_date=None, end_date=None, rebalance_frequency=None):
        """
        Exécute le backtest pour toutes les stratégies.
        
        Args:
            start_date (str, optional): Date de début au format 'YYYY-MM-DD'. Par défaut à 10 ans avant aujourd'hui.
            end_date (str, optional): Date de fin au format 'YYYY-MM-DD'. Par défaut à aujourd'hui.
            rebalance_frequency (str, optional): Fréquence de rééquilibrage à appliquer à toutes les stratégies.
                                                Par défaut, utilise la fréquence définie pour chaque stratégie.
        
        Returns:
            dict: Résultats du backtest pour chaque stratégie.
        """
        if not start_date:
            start_date = (datetime.now() - timedelta(days=3650)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        print(f"Chargement des données obligataires pour la période {start_date} à {end_date}...")
        # Charger les données des ETF obligataires
        etf_data = self.data_loader.load_bond_etfs(start_date=start_date, end_date=end_date)
        
        # Charger les données de l'indice de référence
        benchmark_data = self.data_loader.load_benchmark_data(
            ticker=self.benchmark_ticker, 
            start_date=start_date, 
            end_date=end_date
        )
        
        # S'assurer que l'indice de référence est dans les données des ETF
        if self.benchmark_ticker not in etf_data.columns:
            etf_data[self.benchmark_ticker] = benchmark_data['Adj Close']
            
        # Exécuter le backtest pour chaque stratégie
        print(f"Exécution du backtest pour {len(self.strategies)} stratégies...")
        self.results = {}
        
        for strategy in tqdm(self.strategies, desc="Backtesting des stratégies"):
            # Appliquer la fréquence de rééquilibrage si spécifiée
            if rebalance_frequency:
                original_frequency = strategy.rebalance_frequency
                strategy.rebalance_frequency = rebalance_frequency
                
            # Exécuter le backtest
            strategy_result = strategy.backtest(etf_data, start_date=start_date, end_date=end_date)
            
            # Stocker les résultats
            self.results[strategy.name] = {
                'portfolio': strategy.portfolio.copy(),
                'performance': strategy.performance,
                'transactions': strategy.transactions.copy()
            }
            
            # Restaurer la fréquence de rééquilibrage d'origine si nécessaire
            if rebalance_frequency:
                strategy.rebalance_frequency = original_frequency
                
        # Exécuter le backtest pour l'indice de référence (buy and hold)
        print(f"Exécution du backtest pour l'indice de référence {self.benchmark_ticker}...")
        benchmark_portfolio = pd.DataFrame(index=etf_data.index)
        
        # Normaliser les prix de l'indice de référence
        initial_price = etf_data.loc[etf_data.index[0], self.benchmark_ticker]
        benchmark_portfolio['value'] = etf_data[self.benchmark_ticker] / initial_price * 1000000
        
        # Calculer les rendements
        benchmark_portfolio['returns'] = benchmark_portfolio['value'].pct_change()
        
        # Calculer les performances de l'indice de référence
        benchmark_returns = benchmark_portfolio['returns'].iloc[1:]
        years = len(benchmark_returns) / 252
        
        total_return = (benchmark_portfolio['value'].iloc[-1] / benchmark_portfolio['value'].iloc[0]) - 1
        annualized_return = (1 + total_return) ** (1 / years) - 1
        volatility = benchmark_returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - 0.02) / volatility
        
        # Calculer le drawdown
        benchmark_portfolio['cumulative_return'] = (1 + benchmark_portfolio['returns']).cumprod()
        benchmark_portfolio['running_max'] = benchmark_portfolio['cumulative_return'].cummax()
        benchmark_portfolio['drawdown'] = (benchmark_portfolio['cumulative_return'] / benchmark_portfolio['running_max']) - 1
        max_drawdown = benchmark_portfolio['drawdown'].min()
        
        # Stocker les résultats pour l'indice de référence
        self.results[f"Benchmark ({self.benchmark_ticker})"] = {
            'portfolio': benchmark_portfolio,
            'performance': {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'final_value': benchmark_portfolio['value'].iloc[-1],
                'initial_value': benchmark_portfolio['value'].iloc[0]
            },
            'transactions': []
        }
        
        # Compiler les métriques de performance
        self._compile_performance_metrics()
        
        return self.results
    
    def _compile_performance_metrics(self):
        """
        Compile les métriques de performance pour toutes les stratégies et l'indice de référence.
        """
        if not self.results:
            raise ValueError("Backtest must be run before compiling performance metrics")
            
        metrics = ['total_return', 'annualized_return', 'volatility', 'sharpe_ratio', 'max_drawdown']
        self.performance_metrics = pd.DataFrame(index=self.results.keys(), columns=metrics)
        
        for strategy_name, result in self.results.items():
            for metric in metrics:
                self.performance_metrics.loc[strategy_name, metric] = result['performance'][metric]
                
        return self.performance_metrics
    
    def plot_performance(self, figsize=(12, 8)):
        """
        Affiche les performances des stratégies et de l'indice de référence.
        
        Args:
            figsize (tuple, optional): Taille de la figure. Par défaut (12, 8).
            
        Returns:
            matplotlib.figure.Figure: Figure contenant les graphiques.
        """
        if not self.results:
            raise ValueError("Backtest must be run before plotting performance")
            
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Graphique 1: Valeur du portefeuille
        ax1 = axes[0, 0]
        for strategy_name, result in self.results.items():
            portfolio = result['portfolio']
            ax1.plot(portfolio.index, portfolio['value'], label=strategy_name)
            
        ax1.set_title('Évolution de la Valeur du Portefeuille')
        ax1.set_ylabel('Valeur ($)')
        ax1.legend()
        ax1.grid(True)
        
        # Graphique 2: Drawdown
        ax2 = axes[0, 1]
        for strategy_name, result in self.results.items():
            portfolio = result['portfolio']
            ax2.plot(portfolio.index, portfolio['drawdown'], label=strategy_name)
            
        ax2.set_title('Drawdown')
        ax2.set_ylabel('Drawdown (%)')
        ax2.legend()
        ax2.grid(True)
        
        # Graphique 3: Rendements cumulatifs
        ax3 = axes[1, 0]
        for strategy_name, result in self.results.items():
            portfolio = result['portfolio']
            if 'cumulative_return' in portfolio.columns:
                ax3.plot(portfolio.index, portfolio['cumulative_return'], label=strategy_name)
            
        ax3.set_title('Rendements Cumulatifs')
        ax3.set_ylabel('Rendement Cumulatif')
        ax3.legend()
        ax3.grid(True)
        
        # Graphique 4: Métriques de performance
        ax4 = axes[1, 1]
        self.performance_metrics[['annualized_return', 'volatility', 'sharpe_ratio']].plot(kind='bar', ax=ax4)
        ax4.set_title('Métriques de Performance')
        ax4.set_ylabel('Valeur')
        ax4.grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_drawdown_comparison(self, figsize=(10, 6)):
        """
        Affiche une comparaison des drawdowns entre les stratégies et l'indice de référence.
        
        Args:
            figsize (tuple, optional): Taille de la figure. Par défaut (10, 6).
            
        Returns:
            matplotlib.figure.Figure: Figure contenant le graphique de drawdown.
        """
        if not self.results:
            raise ValueError("Backtest must be run before plotting drawdown comparison")
            
        fig, ax = plt.subplots(figsize=figsize)
        
        for strategy_name, result in self.results.items():
            portfolio = result['portfolio']
            ax.fill_between(
                portfolio.index, 
                0, 
                portfolio['drawdown'] * 100,  # Convertir en pourcentage
                alpha=0.3, 
                label=strategy_name
            )
            
        ax.set_title('Comparaison des Drawdowns')
        ax.set_ylabel('Drawdown (%)')
        ax.legend()
        ax.grid(True)
        
        return fig
    
    def plot_rolling_returns(self, window=252, figsize=(10, 6)):
        """
        Affiche les rendements glissants pour les stratégies et l'indice de référence.
        
        Args:
            window (int, optional): Taille de la fenêtre glissante en jours. Par défaut 252 (1 an).
            figsize (tuple, optional): Taille de la figure. Par défaut (10, 6).
            
        Returns:
            matplotlib.figure.Figure: Figure contenant le graphique des rendements glissants.
        """
        if not self.results:
            raise ValueError("Backtest must be run before plotting rolling returns")
            
        fig, ax = plt.subplots(figsize=figsize)
        
        for strategy_name, result in self.results.items():
            portfolio = result['portfolio']
            returns = portfolio['returns'].iloc[1:]  # Exclure la première valeur qui est NaN
            
            # Calculer les rendements glissants annualisés
            rolling_returns = returns.rolling(window=window).mean() * 252 * 100  # Annualisé et en pourcentage
            
            ax.plot(rolling_returns.index, rolling_returns, label=strategy_name)
            
        ax.set_title(f'Rendements Glissants sur {window} jours (Annualisés)')
        ax.set_ylabel('Rendement Annualisé (%)')
        ax.legend()
        ax.grid(True)
        
        return fig
    
    def plot_monthly_returns_heatmap(self, strategy_name, figsize=(12, 8)):
        """
        Affiche une heatmap des rendements mensuels pour une stratégie donnée.
        
        Args:
            strategy_name (str): Nom de la stratégie.
            figsize (tuple, optional): Taille de la figure. Par défaut (12, 8).
            
        Returns:
            matplotlib.figure.Figure: Figure contenant la heatmap.
        """
        if not self.results or strategy_name not in self.results:
            raise ValueError(f"Backtest must be run for strategy '{strategy_name}' before plotting monthly returns")
            
        # Obtenir les rendements pour la stratégie
        portfolio = self.results[strategy_name]['portfolio']
        returns = portfolio['returns'].iloc[1:]  # Exclure la première valeur qui est NaN
        
        # Convertir en rendements mensuels
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
        
        # Créer un DataFrame avec les rendements mensuels par année et mois
        monthly_returns_df = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month,
            'Return': monthly_returns.values
        })
        
        # Pivoter pour créer la heatmap
        pivot_table = monthly_returns_df.pivot('Year', 'Month', 'Return')
        
        # Noms des mois
        month_names = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 'Juil', 'Août', 'Sep', 'Oct', 'Nov', 'Déc']
        pivot_table.columns = [month_names[i-1] for i in pivot_table.columns]
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            pivot_table, 
            annot=True, 
            fmt=".2f", 
            cmap="RdYlGn", 
            center=0, 
            linewidths=1, 
            ax=ax
        )
        
        ax.set_title(f'Rendements Mensuels (%) - {strategy_name}')
        
        return fig
    
    def get_performance_summary(self):
        """
        Génère un résumé des performances pour toutes les stratégies.
        
        Returns:
            pandas.DataFrame: Tableau récapitulatif des performances.
        """
        if not self.performance_metrics.empty:
            # Formater les valeurs pour l'affichage
            summary = self.performance_metrics.copy()
            summary['total_return'] = summary['total_return'].apply(lambda x: f"{x*100:.2f}%")
            summary['annualized_return'] = summary['annualized_return'].apply(lambda x: f"{x*100:.2f}%")
            summary['volatility'] = summary['volatility'].apply(lambda x: f"{x*100:.2f}%")
            summary['max_drawdown'] = summary['max_drawdown'].apply(lambda x: f"{x*100:.2f}%")
            
            return summary
        else:
            raise ValueError("Backtest must be run before getting performance summary")
    
    def analyze_drawdowns(self, min_length=5, min_depth=0.05):
        """
        Analyse les périodes de drawdown significatives pour chaque stratégie.
        
        Args:
            min_length (int, optional): Durée minimale du drawdown en jours. Par défaut 5.
            min_depth (float, optional): Profondeur minimale du drawdown (0.05 = 5%). Par défaut 0.05.
            
        Returns:
            dict: Dictionnaire contenant les analyses de drawdown par stratégie.
        """
        if not self.results:
            raise ValueError("Backtest must be run before analyzing drawdowns")
            
        drawdown_analysis = {}
        
        for strategy_name, result in self.results.items():
            portfolio = result['portfolio']
            
            # Identifier les périodes de drawdown
            in_drawdown = False
            drawdown_periods = []
            current_period = {}
            
            for date, row in portfolio.iterrows():
                drawdown = row['drawdown']
                
                if drawdown < -min_depth and not in_drawdown:
                    # Début d'une période de drawdown
                    in_drawdown = True
                    current_period = {
                        'start_date': date,
                        'peak_date': date,
                        'peak_value': row['value'],
                        'max_drawdown': drawdown
                    }
                elif in_drawdown:
                    # Mise à jour de la période en cours
                    if drawdown < current_period['max_drawdown']:
                        current_period['max_drawdown'] = drawdown
                        current_period['trough_date'] = date
                        current_period['trough_value'] = row['value']
                        
                    if drawdown >= -0.01:  # Considérer un drawdown terminé quand on revient à -1% du sommet
                        # Fin de la période de drawdown
                        in_drawdown = False
                        current_period['end_date'] = date
                        current_period['duration'] = (date - current_period['start_date']).days
                        current_period['recovery_duration'] = (date - current_period['trough_date']).days if 'trough_date' in current_period else 0
                        
                        # Ajouter la période si elle dépasse la durée minimale
                        if current_period['duration'] >= min_length:
                            drawdown_periods.append(current_period)
                            
            # Filtrer pour inclure uniquement les drawdowns suffisamment profonds
            significant_drawdowns = [period for period in drawdown_periods if period['max_drawdown'] <= -min_depth]
            
            # Trier par profondeur de drawdown
            significant_drawdowns.sort(key=lambda x: x['max_drawdown'])
            
            drawdown_analysis[strategy_name] = significant_drawdowns
            
        return drawdown_analysis
            
    def analyze_performance_by_economic_cycle(self, economic_data):
        """
        Analyse la performance des stratégies par cycle économique.
        
        Args:
            economic_data (pandas.DataFrame): DataFrame contenant des indicateurs du cycle économique.
                Doit avoir une colonne 'cycle' avec les valeurs ['expansion', 'peak', 'contraction', 'trough'].
                
        Returns:
            pandas.DataFrame: Performance par cycle économique pour chaque stratégie.
        """
        if not self.results:
            raise ValueError("Backtest must be run before analyzing performance by economic cycle")
            
        # S'assurer que l'index est une datetime
        economic_data = economic_data.copy()
        if not isinstance(economic_data.index, pd.DatetimeIndex):
            economic_data.index = pd.to_datetime(economic_data.index)
            
        # Vérifier que la colonne 'cycle' existe
        if 'cycle' not in economic_data.columns:
            raise ValueError("economic_data must have a 'cycle' column")
            
        # Initialiser le DataFrame de résultats
        cycles = ['expansion', 'peak', 'contraction', 'trough']
        metrics = ['return', 'volatility', 'sharpe_ratio']
        
        multi_index = pd.MultiIndex.from_product([cycles, metrics], names=['cycle', 'metric'])
        results = pd.DataFrame(index=self.results.keys(), columns=multi_index)
        
        # Analyser chaque stratégie
        for strategy_name, result in self.results.items():
            portfolio = result['portfolio']
            returns = portfolio['returns'].iloc[1:]  # Exclure la première valeur qui est NaN
            
            # Aligner les dates des rendements avec les données économiques
            aligned_data = pd.merge_asof(
                returns.reset_index(), 
                economic_data['cycle'].reset_index(), 
                left_on='index', 
                right_on='index', 
                direction='forward'
            ).set_index('index')
            
            # Calculer les métriques pour chaque cycle
            for cycle in cycles:
                cycle_returns = aligned_data[aligned_data['cycle'] == cycle]['returns']
                
                if len(cycle_returns) > 0:
                    # Rendement total
                    total_return = (1 + cycle_returns).prod() - 1
                    
                    # Volatilité annualisée
                    volatility = cycle_returns.std() * np.sqrt(252)
                    
                    # Ratio de Sharpe
                    sharpe_ratio = (total_return / len(cycle_returns) * 252 - 0.02) / volatility if volatility > 0 else 0
                    
                    results.loc[strategy_name, (cycle, 'return')] = total_return
                    results.loc[strategy_name, (cycle, 'volatility')] = volatility
                    results.loc[strategy_name, (cycle, 'sharpe_ratio')] = sharpe_ratio
                    
        return results
