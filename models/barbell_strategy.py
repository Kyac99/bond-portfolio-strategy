"""
Module pour la stratégie obligataire "Barbell".
"""

import pandas as pd
import numpy as np
from models.base_strategy import BondStrategy


class BarbellStrategy(BondStrategy):
    """
    Implémentation de la stratégie obligataire "Barbell".
    
    La stratégie Barbell consiste à investir aux deux extrémités de la courbe des taux :
    - Une partie en obligations à court terme pour la liquidité et la protection contre la hausse des taux
    - Une partie en obligations à long terme pour capturer les rendements plus élevés
    
    Cette approche évite délibérément les échéances intermédiaires, d'où son nom qui évoque 
    la forme d'une haltère (barbell en anglais).
    """
    
    def __init__(self, short_weight=0.5, long_weight=0.5, short_term_max=3, long_term_min=10, **kwargs):
        """
        Initialise une stratégie Barbell.
        
        Args:
            short_weight (float, optional): Poids à allouer aux obligations à court terme. Par défaut 0.5.
            long_weight (float, optional): Poids à allouer aux obligations à long terme. Par défaut 0.5.
            short_term_max (int, optional): Maturité maximale pour les obligations à court terme (en années). Par défaut 3.
            long_term_min (int, optional): Maturité minimale pour les obligations à long terme (en années). Par défaut 10.
            **kwargs: Arguments supplémentaires à passer à la classe parent.
        """
        super().__init__(name="Barbell Strategy", **kwargs)
        
        # Normaliser les poids
        total = short_weight + long_weight
        self.short_weight = short_weight / total
        self.long_weight = long_weight / total
        
        self.short_term_max = short_term_max
        self.long_term_min = long_term_min
        
        # ETFs à utiliser pour chaque segment
        self.short_term_etfs = ["SHY", "SCHO"]  # iShares/Schwab 1-3 Year Treasury ETFs
        self.long_term_etfs = ["TLT", "EDV"]    # iShares 20+ Year Treasury & Vanguard Extended Duration Treasury ETFs
        
    def generate_weights(self, data, date):
        """
        Génère les poids pour chaque ETF obligataire à une date donnée,
        en fonction de la stratégie barbell.
        
        Args:
            data (pandas.DataFrame): Données des ETF obligataires.
            date (datetime): Date pour laquelle générer les poids.
            
        Returns:
            pandas.Series: Poids à allouer à chaque ETF.
        """
        # Si la date n'est pas dans les données, prendre la date la plus proche
        if date not in data.index:
            closest_date = data.index[data.index.get_indexer([date], method='nearest')[0]]
            date = closest_date
            
        # Initialiser les poids à zéro pour tous les ETF
        available_etfs = []
        
        # Vérifier quels ETFs à court terme sont disponibles
        short_available = [etf for etf in self.short_term_etfs if etf in data.columns]
        if short_available:
            available_etfs.extend(short_available)
            
        # Vérifier quels ETFs à long terme sont disponibles
        long_available = [etf for etf in self.long_term_etfs if etf in data.columns]
        if long_available:
            available_etfs.extend(long_available)
            
        if not available_etfs:
            raise ValueError("Aucun ETF disponible pour la stratégie Barbell")
            
        # Créer le dictionnaire de poids
        weights = {}
        
        # Distribuer le poids court terme entre les ETFs courts disponibles
        if short_available:
            short_etf_weight = self.short_weight / len(short_available)
            for etf in short_available:
                weights[etf] = short_etf_weight
                
        # Distribuer le poids long terme entre les ETFs longs disponibles
        if long_available:
            long_etf_weight = self.long_weight / len(long_available)
            for etf in long_available:
                weights[etf] = long_etf_weight
                
        # Si un segment est manquant, redistribuer son poids à l'autre segment
        if not short_available and long_available:
            long_etf_weight = 1.0 / len(long_available)
            for etf in long_available:
                weights[etf] = long_etf_weight
                
        if not long_available and short_available:
            short_etf_weight = 1.0 / len(short_available)
            for etf in short_available:
                weights[etf] = short_etf_weight
                
        return pd.Series(weights)
    
    def adjust_for_volatility(self, data, lookback_period=60):
        """
        Ajuste les poids en fonction de la volatilité récente des différents segments.
        
        Diminue l'exposition aux segments plus volatils et augmente l'exposition
        aux segments moins volatils.
        
        Args:
            data (pandas.DataFrame): Données historiques des ETF.
            lookback_period (int, optional): Période de lookback en jours. Par défaut 60.
            
        Returns:
            tuple: Nouveaux poids ajustés (short_weight, long_weight).
        """
        # Calculer les rendements
        returns = data.pct_change().dropna()
        
        # Limiter à la période de lookback
        if len(returns) > lookback_period:
            returns = returns.iloc[-lookback_period:]
            
        # Calculer la volatilité pour chaque segment
        short_returns = returns[self.short_term_etfs].mean(axis=1)
        long_returns = returns[self.long_term_etfs].mean(axis=1)
        
        short_vol = short_returns.std() * np.sqrt(252)  # Annualisée
        long_vol = long_returns.std() * np.sqrt(252)    # Annualisée
        
        # Ajuster les poids inversement proportionnels à la volatilité
        if short_vol == 0 or long_vol == 0:  # Éviter division par zéro
            return (self.short_weight, self.long_weight)
            
        short_weight_adj = 1 / short_vol
        long_weight_adj = 1 / long_vol
        
        # Normaliser
        total = short_weight_adj + long_weight_adj
        short_weight_adj = short_weight_adj / total
        long_weight_adj = long_weight_adj / total
        
        return (short_weight_adj, long_weight_adj)
    
    def adjust_for_yield_curve(self, yield_curve, inversion_threshold=-0.1):
        """
        Ajuste les poids en fonction de la forme de la courbe des taux.
        
        En cas d'inversion de la courbe, privilégie les obligations à court terme.
        En cas de courbe pentue, privilégie les obligations à long terme.
        
        Args:
            yield_curve (pandas.Series): Courbe des taux avec les rendements par maturité.
            inversion_threshold (float, optional): Seuil de pente pour considérer la courbe comme inversée.
                                                 Par défaut -0.1.
            
        Returns:
            tuple: Nouveaux poids ajustés (short_weight, long_weight).
        """
        # Calculer la pente de la courbe des taux
        if '30Y' in yield_curve and '3M' in yield_curve:
            slope = yield_curve['30Y'] - yield_curve['3M']
        elif '10Y' in yield_curve and '3M' in yield_curve:
            slope = yield_curve['10Y'] - yield_curve['3M']
        else:
            # Si nous n'avons pas les points nécessaires, utiliser les poids par défaut
            return (self.short_weight, self.long_weight)
            
        # Ajuster les poids en fonction de la pente
        if slope < inversion_threshold:  # Courbe inversée
            # Favoriser court terme
            short_weight_adj = 0.8
            long_weight_adj = 0.2
        elif slope > 2.0:  # Courbe très pentue
            # Favoriser long terme
            short_weight_adj = 0.3
            long_weight_adj = 0.7
        else:  # Courbe normale
            # Équilibrer
            short_weight_adj = 0.5
            long_weight_adj = 0.5
            
        return (short_weight_adj, long_weight_adj)
