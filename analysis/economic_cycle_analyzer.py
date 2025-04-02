"""
Module pour l'analyse des cycles économiques et leur impact sur les obligations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.signal import find_peaks


class EconomicCycleAnalyzer:
    """
    Classe pour analyser les cycles économiques et leur impact sur les stratégies obligataires.
    """
    
    def __init__(self, data_loader):
        """
        Initialise l'analyseur de cycles économiques.
        
        Args:
            data_loader: Instance de DataLoader pour charger les données économiques.
        """
        self.data_loader = data_loader
        self.economic_data = None
        self.cycles = None
        
    def load_economic_data(self, start_date=None, end_date=None):
        """
        Charge les données économiques pour l'analyse.
        
        Args:
            start_date (str, optional): Date de début au format 'YYYY-MM-DD'. Par défaut à None.
            end_date (str, optional): Date de fin au format 'YYYY-MM-DD'. Par défaut à None.
            
        Returns:
            pandas.DataFrame: Données économiques.
        """
        self.economic_data = self.data_loader.load_economic_indicators(start_date=start_date, end_date=end_date)
        return self.economic_data
    
    def identify_cycles(self, method='pca', window=12, smoothing=3):
        """
        Identifie les phases du cycle économique dans les données.
        
        Args:
            method (str, optional): Méthode d'identification ('pca', 'gdp', 'composite'). Par défaut 'pca'.
            window (int, optional): Fenêtre pour le calcul des tendances. Par défaut 12 (mois).
            smoothing (int, optional): Fenêtre de lissage pour les indicateurs. Par défaut 3 (mois).
            
        Returns:
            pandas.DataFrame: DataFrame avec les phases du cycle identifiées.
        """
        if self.economic_data is None:
            raise ValueError("Economic data must be loaded before identifying cycles")
            
        # Créer une copie des données
        data = self.economic_data.copy()
        
        # Appliquer un lissage si nécessaire
        if smoothing > 0:
            data = data.rolling(window=smoothing, min_periods=1).mean()
            
        if method == 'pca':
            # Utiliser l'analyse en composantes principales pour extraire le cycle économique
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data.dropna())
            
            # Appliquer PCA
            pca = PCA(n_components=1)
            principal_component = pca.fit_transform(scaled_data).flatten()
            
            # Reconstruire le DataFrame
            cycle_indicator = pd.Series(principal_component, index=data.dropna().index)
            
            # Calculer la dérivée pour déterminer la direction
            cycle_slope = cycle_indicator.diff().rolling(window=window).mean()
            
            # Définir les phases du cycle
            phases = pd.Series(index=cycle_indicator.index)
            phases[cycle_indicator > 0] = "expansion" if cycle_slope.iloc[-1] > 0 else "peak"
            phases[cycle_indicator <= 0] = "contraction" if cycle_slope.iloc[-1] < 0 else "trough"
            
        elif method == 'gdp':
            # Utiliser le PIB comme indicateur principal
            if 'GDP' not in data.columns:
                raise ValueError("GDP data is required for the 'gdp' method")
                
            # Calculer le taux de croissance du PIB
            gdp_growth = data['GDP'].pct_change(periods=4)  # Croissance annuelle
            
            # Lisser la croissance
            gdp_growth_smoothed = gdp_growth.rolling(window=smoothing, min_periods=1).mean()
            
            # Définir les phases selon la croissance et son évolution
            phases = pd.Series(index=gdp_growth_smoothed.index)
            
            # Calcul de la dérivée (accélération/décélération)
            gdp_acceleration = gdp_growth_smoothed.diff().rolling(window=window // 2, min_periods=1).mean()
            
            # Définir les phases
            phases[(gdp_growth_smoothed > 0) & (gdp_acceleration > 0)] = "expansion"
            phases[(gdp_growth_smoothed > 0) & (gdp_acceleration <= 0)] = "peak"
            phases[(gdp_growth_smoothed <= 0) & (gdp_acceleration <= 0)] = "contraction"
            phases[(gdp_growth_smoothed <= 0) & (gdp_acceleration > 0)] = "trough"
            
        elif method == 'composite':
            # Utiliser une combinaison d'indicateurs
            # Calculer les variations pour chaque indicateur
            changes = data.pct_change(periods=window)
            
            # Créer un indicateur composite en pondérant les différentes mesures
            weights = {
                'GDP': 0.4,
                'Industrial_Production': 0.2,
                'Unemployment': -0.2,  # Inverser car une baisse du chômage est positive
                'CPI': -0.1,  # Inverser car une inflation élevée peut être négative
                'Retail_Sales': 0.1
            }
            
            # Ne prendre que les indicateurs disponibles
            available_indicators = [ind for ind in weights.keys() if ind in changes.columns]
            
            # Ajuster les poids
            total_weight = sum(abs(weights[ind]) for ind in available_indicators)
            adjusted_weights = {ind: weights[ind] / total_weight for ind in available_indicators}
            
            # Calculer l'indicateur composite
            composite = pd.Series(0, index=changes.index)
            for indicator, weight in adjusted_weights.items():
                if indicator == 'Unemployment':
                    # Inverser pour que la baisse du chômage soit positive
                    composite -= changes[indicator] * abs(weight)
                else:
                    composite += changes[indicator] * weight
                    
            # Lisser l'indicateur composite
            composite_smoothed = composite.rolling(window=smoothing, min_periods=1).mean()
            
            # Calculer la dérivée
            composite_slope = composite_smoothed.diff().rolling(window=window // 2, min_periods=1).mean()
            
            # Définir les phases
            phases = pd.Series(index=composite_smoothed.index)
            phases[(composite_smoothed > 0) & (composite_slope > 0)] = "expansion"
            phases[(composite_smoothed > 0) & (composite_slope <= 0)] = "peak"
            phases[(composite_smoothed <= 0) & (composite_slope <= 0)] = "contraction"
            phases[(composite_smoothed <= 0) & (composite_slope > 0)] = "trough"
            
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'pca', 'gdp', or 'composite'.")
            
        # Créer le DataFrame final
        self.cycles = pd.DataFrame(phases, columns=['cycle'])
        
        # Remplir les valeurs manquantes par propagation
        self.cycles = self.cycles.ffill().bfill()
        
        return self.cycles
    
    def analyze_bond_performance_by_cycle(self, bond_data):
        """
        Analyse la performance des obligations par cycle économique.
        
        Args:
            bond_data (pandas.DataFrame): Données des ETF obligataires.
            
        Returns:
            pandas.DataFrame: Performance des obligations par cycle économique.
        """
        if self.cycles is None:
            raise ValueError("Economic cycles must be identified before analyzing bond performance")
            
        # Ajuster la fréquence des données de cycle à celle des données obligataires
        daily_cycles = self.cycles.resample('D').ffill()
        
        # Fusionner les données
        merged_data = pd.merge_asof(
            bond_data.reset_index(), 
            daily_cycles.reset_index(), 
            left_on='index', 
            right_on='index', 
            direction='forward'
        ).set_index('index')
        
        # Calculer les rendements
        returns = merged_data.pct_change().dropna()
        
        # Ajouter la colonne de cycle
        returns['cycle'] = merged_data['cycle']
        
        # Analyser les performances par cycle
        performance = {}
        
        for cycle in ['expansion', 'peak', 'contraction', 'trough']:
            cycle_returns = returns[returns['cycle'] == cycle].drop(columns=['cycle'])
            
            if not cycle_returns.empty:
                # Calculer les métriques
                annualized_returns = cycle_returns.mean() * 252
                volatility = cycle_returns.std() * np.sqrt(252)
                sharpe_ratio = annualized_returns / volatility
                
                performance[cycle] = {
                    'annualized_returns': annualized_returns,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'duration': len(cycle_returns)
                }
                
        return performance
    
    def plot_cycles(self, figsize=(12, 8)):
        """
        Visualise les cycles économiques identifiés.
        
        Args:
            figsize (tuple, optional): Taille de la figure. Par défaut (12, 8).
            
        Returns:
            matplotlib.figure.Figure: Figure contenant les graphiques.
        """
        if self.cycles is None:
            raise ValueError("Economic cycles must be identified before plotting")
            
        # Créer une version numérique des cycles pour faciliter la visualisation
        cycle_map = {
            'expansion': 1,
            'peak': 0.5,
            'contraction': 0,
            'trough': -0.5
        }
        
        numerical_cycles = self.cycles['cycle'].map(cycle_map)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # Graphique 1: Indicateurs économiques
        if self.economic_data is not None:
            # Normaliser les données pour une meilleure visualisation
            scaler = StandardScaler()
            normalized_data = pd.DataFrame(
                scaler.fit_transform(self.economic_data),
                index=self.economic_data.index,
                columns=self.economic_data.columns
            )
            
            for column in normalized_data.columns:
                ax1.plot(normalized_data.index, normalized_data[column], label=column)
                
            ax1.set_title('Indicateurs Économiques Normalisés')
            ax1.legend()
            ax1.grid(True)
            
            # Graphique 2: Phases du cycle économique
            colors = {
                'expansion': 'green',
                'peak': 'orange',
                'contraction': 'red',
                'trough': 'blue'
            }
            
            # Créer une plage continue pour l'arrière-plan
            for cycle in ['expansion', 'peak', 'contraction', 'trough']:
                mask = self.cycles['cycle'] == cycle
                if mask.any():
                    ax2.fill_between(
                        self.cycles.index,
                        -1, 2,
                        where=mask.values,
                        alpha=0.3,
                        color=colors[cycle],
                        label=cycle
                    )
                    
            # Ajouter une ligne représentant la valeur numérique du cycle
            ax2.plot(numerical_cycles.index, numerical_cycles, 'k-', linewidth=2)
            
            ax2.set_title('Phases du Cycle Économique')
            ax2.set_yticks(list(cycle_map.values()))
            ax2.set_yticklabels(list(cycle_map.keys()))
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            
        return fig
    
    def plot_bond_performance_by_cycle(self, bond_data, figsize=(14, 10)):
        """
        Visualise la performance des obligations par cycle économique.
        
        Args:
            bond_data (pandas.DataFrame): Données des ETF obligataires.
            figsize (tuple, optional): Taille de la figure. Par défaut (14, 10).
            
        Returns:
            matplotlib.figure.Figure: Figure contenant les graphiques.
        """
        performance = self.analyze_bond_performance_by_cycle(bond_data)
        
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        
        # Extraire les métriques
        cycles = list(performance.keys())
        etfs = bond_data.columns
        
        # Rendements annualisés
        returns_data = {cycle: [performance[cycle]['annualized_returns'][etf] * 100 for etf in etfs] for cycle in cycles}
        returns_df = pd.DataFrame(returns_data, index=etfs)
        
        # Volatilité
        volatility_data = {cycle: [performance[cycle]['volatility'][etf] * 100 for etf in etfs] for cycle in cycles}
        volatility_df = pd.DataFrame(volatility_data, index=etfs)
        
        # Ratio de Sharpe
        sharpe_data = {cycle: [performance[cycle]['sharpe_ratio'][etf] for etf in etfs] for cycle in cycles}
        sharpe_df = pd.DataFrame(sharpe_data, index=etfs)
        
        # Graphique 1: Rendements annualisés
        returns_df.plot(kind='bar', ax=axes[0], rot=0)
        axes[0].set_title('Rendements Annualisés (%) par Cycle Économique')
        axes[0].set_ylabel('Rendement (%)')
        axes[0].grid(True, axis='y')
        
        # Graphique 2: Volatilité
        volatility_df.plot(kind='bar', ax=axes[1], rot=0)
        axes[1].set_title('Volatilité (%) par Cycle Économique')
        axes[1].set_ylabel('Volatilité (%)')
        axes[1].grid(True, axis='y')
        
        # Graphique 3: Ratio de Sharpe
        sharpe_df.plot(kind='bar', ax=axes[2], rot=0)
        axes[2].set_title('Ratio de Sharpe par Cycle Économique')
        axes[2].set_ylabel('Ratio de Sharpe')
        axes[2].grid(True, axis='y')
        
        plt.tight_layout()
        
        return fig
    
    def identify_yield_curve_regime(self, treasury_yields, window=20):
        """
        Identifie le régime de la courbe des taux (pentue, plate, inversée).
        
        Args:
            treasury_yields (pandas.DataFrame): Rendements des bons du Trésor.
            window (int, optional): Fenêtre pour le calcul des moyennes mobiles. Par défaut 20.
            
        Returns:
            pandas.DataFrame: Régimes de la courbe des taux.
        """
        # Calculer les spreads entre différentes échéances
        if '10Y' in treasury_yields.columns and '3M' in treasury_yields.columns:
            spreads = treasury_yields['10Y'] - treasury_yields['3M']
        elif '10Y' in treasury_yields.columns and '2Y' in treasury_yields.columns:
            spreads = treasury_yields['10Y'] - treasury_yields['2Y']
        else:
            raise ValueError("Treasury yields must contain at least two different maturities")
            
        # Lisser les spreads
        spreads_smoothed = spreads.rolling(window=window).mean()
        
        # Définir les régimes
        regimes = pd.DataFrame(index=spreads.index, columns=['regime'])
        
        # Seuils pour définir les régimes
        steep_threshold = 1.5   # Courbe pentue: spread > 1.5%
        flat_threshold = 0.5    # Courbe plate: 0.5% > spread > 0%
        inverted_threshold = 0  # Courbe inversée: spread < 0%
        
        regimes['regime'] = 'normal'  # Par défaut
        regimes.loc[spreads_smoothed > steep_threshold, 'regime'] = 'steep'
        regimes.loc[(spreads_smoothed <= flat_threshold) & (spreads_smoothed > inverted_threshold), 'regime'] = 'flat'
        regimes.loc[spreads_smoothed <= inverted_threshold, 'regime'] = 'inverted'
        
        return regimes
    
    def analyze_bond_performance_by_yield_curve(self, bond_data, treasury_yields):
        """
        Analyse la performance des obligations selon le régime de la courbe des taux.
        
        Args:
            bond_data (pandas.DataFrame): Données des ETF obligataires.
            treasury_yields (pandas.DataFrame): Rendements des bons du Trésor.
            
        Returns:
            pandas.DataFrame: Performance des obligations par régime de courbe des taux.
        """
        # Identifier les régimes de la courbe des taux
        yield_regimes = self.identify_yield_curve_regime(treasury_yields)
        
        # Fusionner les données
        merged_data = pd.merge_asof(
            bond_data.reset_index(), 
            yield_regimes.reset_index(), 
            left_on='index', 
            right_on='index', 
            direction='forward'
        ).set_index('index')
        
        # Calculer les rendements
        returns = merged_data.iloc[:, :-1].pct_change().dropna()
        
        # Ajouter la colonne de régime
        returns['regime'] = merged_data['regime']
        
        # Analyser les performances par régime
        performance = {}
        
        for regime in ['steep', 'normal', 'flat', 'inverted']:
            regime_returns = returns[returns['regime'] == regime].drop(columns=['regime'])
            
            if not regime_returns.empty:
                # Calculer les métriques
                annualized_returns = regime_returns.mean() * 252
                volatility = regime_returns.std() * np.sqrt(252)
                sharpe_ratio = annualized_returns / volatility
                
                performance[regime] = {
                    'annualized_returns': annualized_returns,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'duration': len(regime_returns)
                }
                
        return performance
    
    def generate_combined_analysis(self, bond_data, treasury_yields):
        """
        Génère une analyse combinée des cycles économiques et des régimes de courbe des taux.
        
        Args:
            bond_data (pandas.DataFrame): Données des ETF obligataires.
            treasury_yields (pandas.DataFrame): Rendements des bons du Trésor.
            
        Returns:
            dict: Analyse combinée avec des recommandations de stratégie.
        """
        # S'assurer que les cycles ont été identifiés
        if self.cycles is None:
            raise ValueError("Economic cycles must be identified before generating combined analysis")
            
        # Identifier les régimes de la courbe des taux
        yield_regimes = self.identify_yield_curve_regime(treasury_yields)
        
        # Fusionner les données de cycle et de régime
        combined = pd.merge_asof(
            self.cycles.reset_index(), 
            yield_regimes.reset_index(), 
            left_on='index', 
            right_on='index', 
            direction='forward'
        ).set_index('index')
        
        # Analyser les performances des obligations dans les différentes combinaisons
        merged_data = pd.merge_asof(
            bond_data.reset_index(), 
            combined.reset_index(), 
            left_on='index', 
            right_on='index', 
            direction='forward'
        ).set_index('index')
        
        # Calculer les rendements
        bond_cols = bond_data.columns
        returns = merged_data[bond_cols].pct_change().dropna()
        
        # Ajouter les colonnes de cycle et de régime
        returns['cycle'] = merged_data['cycle']
        returns['regime'] = merged_data['regime']
        
        # Analyser les performances par combinaison cycle-régime
        performance = {}
        
        for cycle in ['expansion', 'peak', 'contraction', 'trough']:
            for regime in ['steep', 'normal', 'flat', 'inverted']:
                combo_returns = returns[(returns['cycle'] == cycle) & (returns['regime'] == regime)]
                combo_returns = combo_returns[bond_cols]  # Garder uniquement les colonnes des obligations
                
                if len(combo_returns) > 20:  # Au moins 20 observations pour être significatif
                    # Calculer les métriques
                    annualized_returns = combo_returns.mean() * 252
                    volatility = combo_returns.std() * np.sqrt(252)
                    sharpe_ratio = annualized_returns / volatility
                    
                    # Trouver la meilleure stratégie (ETF avec le meilleur Sharpe ratio)
                    best_etf = sharpe_ratio.idxmax()
                    
                    performance[f"{cycle}_{regime}"] = {
                        'annualized_returns': annualized_returns,
                        'volatility': volatility,
                        'sharpe_ratio': sharpe_ratio,
                        'duration': len(combo_returns),
                        'best_etf': best_etf,
                        'best_sharpe': sharpe_ratio[best_etf]
                    }
                    
        # Définir les recommandations de stratégie
        recommendations = {}
        
        # Laddering, Barbell, Bullet
        strategy_recommendations = {
            'expansion_steep': 'Laddering - Favoriser une exposition diversifiée à la courbe des taux en pleine croissance',
            'expansion_normal': 'Laddering - Maintenir une diversification avec un léger biais vers les maturités intermédiaires',
            'expansion_flat': 'Barbell - Se positionner aux extrémités pour capturer les rendements tout en se protégeant',
            'expansion_inverted': 'Barbell - Privilégier les maturités courtes avec une petite exposition aux longues',
            'peak_steep': 'Laddering - Rester diversifié mais commencer à diminuer les durations',
            'peak_normal': 'Bullet - Concentrer sur les maturités intermédiaires pour un bon rapport rendement/risque',
            'peak_flat': 'Barbell - Surpondérer le court terme en prévision d'un possible ralentissement',
            'peak_inverted': 'Bullet - Concentration sur le court terme pour se protéger',
            'contraction_steep': 'Bullet - Cibler les maturités longues pour bénéficier de la baisse des taux',
            'contraction_normal': 'Bullet - Maintenir l'exposition aux maturités moyennes à longues',
            'contraction_flat': 'Barbell - Équilibrer entre court terme pour la sécurité et long terme pour la performance',
            'contraction_inverted': 'Barbell - Surpondérer le court terme mais maintenir une exposition au long',
            'trough_steep': 'Bullet - Maximiser la duration pour profiter de la baisse des taux',
            'trough_normal': 'Bullet - Cibler les maturités longues pour verrouiller les rendements',
            'trough_flat': 'Laddering - Commencer à diversifier à travers toute la courbe',
            'trough_inverted': 'Laddering - Se positionner progressivement sur toute la courbe en prévision d'une normalisation'
        }
        
        for combo, strat_recommendation in strategy_recommendations.items():
            if combo in performance:
                recommendations[combo] = {
                    'strategy_type': strat_recommendation.split(' - ')[0],
                    'rationale': strat_recommendation.split(' - ')[1],
                    'best_etf': performance[combo]['best_etf'],
                    'expected_return': performance[combo]['annualized_returns'][performance[combo]['best_etf']] * 100,
                    'expected_volatility': performance[combo]['volatility'][performance[combo]['best_etf']] * 100,
                    'sharpe_ratio': performance[combo]['sharpe_ratio'][performance[combo]['best_etf']]
                }
                
        return {
            'performance': performance,
            'recommendations': recommendations,
            'current_cycle': self.cycles['cycle'].iloc[-1],
            'current_regime': yield_regimes['regime'].iloc[-1]
        }
